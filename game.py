import cv2
import numpy as np
import os
import logging

LIMIAR_CORTE = 8.0
logger = logging.getLogger(__name__)

def carregarTemplates(sift_detector, templates_directory):
    template_data = []
    logger.info("Procurando templates em: %s", templates_directory)

    if not os.path.exists(templates_directory):
        logger.error("Diretório de templates '%s' não encontrado.", templates_directory)
        return template_data

    for filename in os.listdir(templates_directory):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            template_path = os.path.join(templates_directory, filename)
            template_name = os.path.splitext(filename)[0] # Nome da carta sem extensão
            template_img = cv2.imread(template_path, 0) # Carregar como escala de cinza

            if template_img is None:
                logger.warning("Não foi possível carregar o template: %s", template_path)
                continue

            kp_template, des_template = sift_detector.detectAndCompute(template_img, None)

            if des_template is not None:
                template_data.append({
                    'name': template_name,
                    'img': template_img,
                    'kp': kp_template,
                    'des': des_template
                })
                logger.info("Template '%s' carregado com %d keypoints.",
                            template_name, len(kp_template))
            else:
                logger.warning("Não foi possível extrair descritores para o template: %s",
                               template_name)

    logger.info("Total de %d templates carregados e processados.", len(template_data))
    return template_data

def encontrar_multiplas_instancias(kp_template, des_template, kp_img, des_img,
                                   template_img, img,
                                   template_name="",
                                   min_matches=8, ratio=0.78,
                                   ransac_thresh=5.5,
                                   caixas_globais=None):
    # --- cross-check ---
    bf_cc = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    mutuos = bf_cc.match(des_template, des_img)
    ids_mutuos = {(m.queryIdx, m.trainIdx) for m in mutuos}

    # --- ratio ---
    bf_knn = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = bf_knn.knnMatch(des_template, des_img, k=2)

    bons = []
    for par in knn:
        if len(par) < 2:
            continue
        m, n = par
        if m.distance < ratio * n.distance and (m.queryIdx, m.trainIdx) in ids_mutuos:
            bons.append(m)

    logger.debug("\t[%s] knn=%d | mutuos=%d | bons(crosscheck+ratio)=%d | min=%d",
                 template_name, len(knn), len(mutuos), len(bons), min_matches)

    if len(bons) < min_matches:
        return 0

    src_all = np.float32([kp_template[m.queryIdx].pt for m in bons])
    dst_all = np.float32([kp_img[m.trainIdx].pt for m in bons])

    indices = list(range(len(bons)))
    caixas_globais = caixas_globais if caixas_globais is not None else []
    n_kp_tpl = len(kp_template)
    LIMIAR_SEGUINTE = 0.25                           # era 0.5
    TAXA_MIN = 0.4                                  # era 0.55      # 50% do primeiro é suficiente

    primeira_inst = True
    
    inliers_primeira = None
    n_inst = 0
    inliers_da_primeira = None
    PALETA = [(0,255,0),(0,0,255),(255,0,0),(0,255,255),(255,0,255),(255,255,0)]

    while len(indices) >= min_matches:
        src_pts = src_all[indices].reshape(-1,1,2)
        dst_pts = dst_all[indices].reshape(-1,1,2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        if H is None or mask is None:
            logger.debug("\t\t[%s] H=None, interrompendo busca.", template_name)
            break

        inliers = mask.ravel().astype(bool)
        n_in = int(inliers.sum())
        if n_in < min_matches:
            logger.debug("\t\t[%s] inliers=%d < min_matches=%d, interrompendo busca.",
                         template_name, n_in, min_matches)
            break

        taxa_inliers = n_in / max(1, len(src_pts))
        if taxa_inliers < TAXA_MIN:
            logger.debug("\t\t[%s] taxa de inliers baixa (%.2f), ignorando.",
                         template_name, taxa_inliers)
            break
        
        if primeira_inst:
            inliers_primeira = n_in
            primeira_inst = False
        else:
            # aqui o IoU já bloqueia o espelhamento 180°
            if n_in < 0.15 * len(bons):

                break
        ok = homografia_plausivel(H, template_img, img.shape, 
                                  debug=True)
        logger.debug("\t\t[%s] geom_ok=%s", template_name, ok)
        if not ok:
            break

        # calcula a caixa projetada
        h_t, w_t = template_img.shape[:2]
        corners = cv2.perspectiveTransform(
            np.float32([[0,0],[0,h_t],[w_t,h_t],[w_t,0]]).reshape(-1,1,2), H
        ).reshape(-1,2)
        box = (corners[:,0].min(), corners[:,1].min(),
               corners[:,0].max(), corners[:,1].max())
        
        # ⬇️ NMS GLOBAL: compara com TUDO já aceito
        sobreposto = False
        
        for (b_old, name_old, inl_old) in caixas_globais:
            i = iou(box, b_old)
            logger.debug("\t\tIoU com %s: %.2f", name_old, i)
            if i > 0.12:
                sobreposto = True
                break

        if sobreposto:
            # remove esses inliers e continua procurando outra carta igual de verdade
            indices = [indices[i] for i in range(len(indices)) if not inliers[i]]
            logger.debug("\t\tBox %s descartado por sobreposição.", box)
            continue
        logger.debug("\t\tNovo box: %s", box)

        caixas_globais.append((box, template_name, n_in))
        cor = PALETA[n_inst % len(PALETA)]
        desenhaContorno_com_H(H, template_img, img, cor,
                              label=f"{template_name}#{n_inst+1}")
        n_inst += 1

        indices = [indices[i] for i in range(len(indices)) if not inliers[i]]

    return n_inst

def homografia_plausivel(H, template_img, scene_shape, debug=False):
    h, w = template_img.shape[:2]
    pts = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, H).reshape(-1,2)

    # ⬇️ teste de convexidade tolerante: só rejeita se MUITO ruim
    area = abs(cv2.contourArea(dst.astype(np.float32)))
    hull = cv2.convexHull(dst.astype(np.float32))
    area_hull = abs(cv2.contourArea(hull))
    solidez = area / (area_hull + 1e-6)
    if debug: logger.debug("\t\tGeometria: solidez=%.2f", solidez)
    if solidez < 0.55:               # era isContourConvex (muito rígido)
        return False

    ratio_area = area / (w * h)
    if not (0.25 < ratio_area < 4.0):   # folga que combinamos
        if debug: logger.debug("\t\tGeometria: area_ratio=%.2f", ratio_area)
        return False

    def side(p, q): return np.linalg.norm(q - p)
    w_det = (side(dst[0], dst[1]) + side(dst[3], dst[2])) / 2
    h_det = (side(dst[0], dst[3]) + side(dst[1], dst[2])) / 2
    if h_det <= 1:
        return False
    ar = (w_det / h_det) / (w / h)
    if not (0.5 < ar < 2.0):
        if debug: logger.debug("\t\tGeometria: ar=%.2f", ar)
        return False

    ih, iw = scene_shape[:2]
    if dst[:,0].min() < -50 or dst[:,0].max() > iw + 50: return False
    if dst[:,1].min() < -50 or dst[:,1].max() > ih + 50: return False
    return True


def desenhaContorno_com_H(H, template_img, img, cor, label=""):
    """Desenha o contorno de UMA instância usando a homografia já calculada."""
    if H is None:
        return
    h, w = template_img.shape[:2]
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)
    cv2.polylines(img, [np.int32(dst)], True, cor, 3)

    # desenha o nome da carta no centro
    if label:
        cx = int(dst[:, 0, 0].mean())
        cy = int(dst[:, 0, 1].mean())
        cv2.putText(img, label, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)

def iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter + 1e-6)

def min_matches_para(n_kp_tpl):
    if n_kp_tpl <= 50:
        return 5      # templates pequenos: aceita menos
    elif n_kp_tpl < 150:
        return 6      # templates médios
    else:
        return 8      # templates grandes: exige mais

def process_frame(frame, sift_detector, template_data):
    img = frame.copy()
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kernel = np.ones((15,15), np.uint8)

    image_lower_hsv = np.array([0, 0, 160])
    image_upper_hsv = np.array([180, 70, 255])

    mask = cv2.inRange(img_hsv, image_lower_hsv, image_upper_hsv)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regioes = [c for c in contornos
                if cv2.contourArea(c) > 15000 and cv2.boundingRect(c)[3] > 100]

    area = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(area, regioes, -1, 255, -1)
    area = cv2.erode(area, np.ones((9, 9), np.uint8))
    area = cv2.dilate(area, np.ones((15,15), np.uint8), iterations=3)

    # Aplica a máscara SEM recortar — mantém o tamanho do frame original
    img_masked = cv2.bitwise_and(img, img, mask=area)

    return cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)

def detectar_cartas(frame, kp_img, des_img, template_data, reverse_order):
    caixas_globais = []   # uma lista só, passada para todos os templates

    for template in (reversed(template_data) if reverse_order else template_data):
        logger.debug("\t%s: %d keypoints, shape=%s", template['name'],
                     len(template['kp']), template['img'].shape)
        n = encontrar_multiplas_instancias(
            template['kp'], template['des'],
            kp_img, des_img,
            template['img'], frame,
            template_name=template['name'],
            min_matches=min_matches_para(len(template['kp'])),
            ratio=0.75,
            ransac_thresh=5.5,
            caixas_globais=caixas_globais,
        )
        if n > 0:
            logger.info("\tCarta '%s': %d instância(s) encontrada(s).",
                        template['name'], n)
        else:
            logger.debug("\tCarta '%s': nenhuma instância encontrada.", template['name'])

def cena_nova(anterior, atual):
    if anterior is None or atual is None:
        return True
    a = cv2.resize(cv2.cvtColor(anterior, cv2.COLOR_BGR2GRAY), (160, 90)).astype(float)
    b = cv2.resize(cv2.cvtColor(atual, cv2.COLOR_BGR2GRAY), (160, 90)).astype(float)
    return np.abs(a - b).mean() > LIMIAR_CORTE

def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="",
        handlers=[
            logging.FileHandler("game.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    cap = cv2.VideoCapture("jogo21.mp4")
    if not cap.isOpened():
        logger.error("Não foi possível acessar o vídeo.")
        return
    
    templates_dir = 'templates/'
    sift = cv2.SIFT_create()

    anterior = None
    rodada = 0
    template_data = carregarTemplates(sift, templates_dir)
    
    # Variáveis para guardar o último resultado processado
    img_display = None

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Não foi possível capturar o próximo frame; encerrando leitura.")
            break

        if cena_nova(anterior, frame):
            rodada += 1
            logger.info("Rodada %d: nova cena detectada.", rodada)

            # Imagem de trabalho (cinza, recortada) — apenas para o SIFT
            img_trabalho = process_frame(frame, sift, template_data)

            # Imagem de exibição — frame original colorido
            img_display = frame.copy()

            # Desenha na imagem colorida, mas usando keypoints da imagem de trabalho
            kp_img, des_img = sift.detectAndCompute(img_trabalho, None)

            if rodada == 7 or rodada == 9:
                detectar_cartas(img_display, kp_img, des_img, template_data, reverse_order=True)
            else:
                detectar_cartas(img_display, kp_img, des_img, template_data, reverse_order=False)
            
        anterior = frame

        # Exibe a imagem colorida com os contornos
        if img_display is not None:
            cv2.imshow('Jogo', cv2.resize(img_display, (1280, 720)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()