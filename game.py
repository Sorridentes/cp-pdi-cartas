import cv2
import numpy as np
import os
import logging

LIMIAR_CORTE = 8.0
logger = logging.getLogger(__name__)

# Mapeamento de valores das cartas para a soma do jogo
VALORES = {'A': 1, 'J': 10, 'Q': 10, 'K': 10, '10': 10, **{str(n): n for n in range(2, 10)}}

def obter_valor_carta(nome_template):
    """Extrai a pontuação da carta com base no nome do template."""
    for chave in sorted(VALORES.keys(), key=len, reverse=True):
        if nome_template.startswith(chave):
            return VALORES[chave]
    return 0

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

    if len(bons) < min_matches:
        return 0

    src_all = np.float32([kp_template[m.queryIdx].pt for m in bons])
    dst_all = np.float32([kp_img[m.trainIdx].pt for m in bons])

    indices = list(range(len(bons)))
    caixas_globais = caixas_globais if caixas_globais is not None else []
    TAXA_MIN = 0.4

    primeira_inst = True
    n_inst = 0

    while len(indices) >= min_matches:
        src_pts = src_all[indices].reshape(-1,1,2)
        dst_pts = dst_all[indices].reshape(-1,1,2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        if H is None or mask is None:
            break

        inliers = mask.ravel().astype(bool)
        n_in = int(inliers.sum())
        if n_in < min_matches:
            break

        taxa_inliers = n_in / max(1, len(src_pts))
        if taxa_inliers < TAXA_MIN:
            break
        
        if primeira_inst:
            primeira_inst = False
        else:
            if n_in < 0.15 * len(bons):
                break

        ok = homografia_plausivel(H, template_img, img.shape, debug=True)
        if not ok:
            break

        # calcula a caixa projetada
        h_t, w_t = template_img.shape[:2]
        corners = cv2.perspectiveTransform(
            np.float32([[0,0],[0,h_t],[w_t,h_t],[w_t,0]]).reshape(-1,1,2), H
        ).reshape(-1,2)
        box = (corners[:,0].min(), corners[:,1].min(),
               corners[:,0].max(), corners[:,1].max())
        
        # NMS GLOBAL: compara com tudo já aceito
        sobreposto = False
        for (b_old, name_old, inl_old) in caixas_globais:
            if iou(box, b_old) > 0.12:
                sobreposto = True
                break

        if sobreposto:
            indices = [indices[i] for i in range(len(indices)) if not inliers[i]]
            continue

        caixas_globais.append((box, template_name, n_in))
        n_inst += 1

        indices = [indices[i] for i in range(len(indices)) if not inliers[i]]

    return n_inst

def homografia_plausivel(H, template_img, scene_shape, debug=False):
    h, w = template_img.shape[:2]
    pts = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, H).reshape(-1,2)

    area = abs(cv2.contourArea(dst.astype(np.float32)))
    hull = cv2.convexHull(dst.astype(np.float32))
    area_hull = abs(cv2.contourArea(hull))
    solidez = area / (area_hull + 1e-6)
    if solidez < 0.55:
        return False

    ratio_area = area / (w * h)
    if not (0.25 < ratio_area < 4.0):
        return False

    def side(p, q): return np.linalg.norm(q - p)
    w_det = (side(dst[0], dst[1]) + side(dst[3], dst[2])) / 2
    h_det = (side(dst[0], dst[3]) + side(dst[1], dst[2])) / 2
    if h_det <= 1:
        return False
    ar = (w_det / h_det) / (w / h)
    if not (0.5 < ar < 2.0):
        return False

    ih, iw = scene_shape[:2]
    if dst[:,0].min() < -50 or dst[:,0].max() > iw + 50: return False
    if dst[:,1].min() < -50 or dst[:,1].max() > ih + 50: return False
    return True

def iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter + 1e-6)

def min_matches_para(n_kp_tpl):
    if n_kp_tpl <= 50:
        return 5
    elif n_kp_tpl < 150:
        return 6
    else:
        return 8

def determinar_vencedor(soma_jogador1, soma_jogador2, tem_cartas=True):
    if not tem_cartas:
        return "Aguardando cartas"
    if soma_jogador1 > 21 and soma_jogador2 > 21:
        return "Ambos passaram de 21"
    if soma_jogador1 > 21:
        return "Vencedor: Jogador 2"
    if soma_jogador2 > 21:
        return "Vencedor: Jogador 1"
    if soma_jogador1 == 21 and soma_jogador2 == 21:
        return "Empate: ambos fizeram 21"
    if soma_jogador1 == 21:
        return "Vencedor: Jogador 1"
    if soma_jogador2 == 21:
        return "Vencedor: Jogador 2"
    if soma_jogador1 == soma_jogador2:
        return "Empate"
    vencedor = 1 if soma_jogador1 > soma_jogador2 else 2
    return f"Vencedor: Jogador {vencedor}"


def desenhar_placar_no_frame(img, caixas_globais):
    altura, largura = img.shape[:2]
    meio = largura // 2
    cartas_por_jogador = [[], []]

    for caixa, nome, _ in caixas_globais:
        centro_x = (caixa[0] + caixa[2]) / 2
        jogador = 0 if centro_x < meio else 1
        cartas_por_jogador[jogador].append(nome)

    somas = [
        sum(obter_valor_carta(nome) for nome in cartas)
        for cartas in cartas_por_jogador
    ]
    resultado = determinar_vencedor(somas[0], somas[1], bool(caixas_globais))

    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (largura - 10, 155), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.line(img, (meio, 0), (meio, altura), (0, 255, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    padding = 24
    largura_texto = max(1, meio - 2 * padding)

    for indice, cartas in enumerate(cartas_por_jogador):
        x = padding if indice == 0 else meio + padding
        nome_jogador = f"Jogador {indice + 1}"
        nomes_exibidos = [nome.split('_', 1)[0] for nome in cartas]
        texto_cartas = "Cartas: " + (" + ".join(nomes_exibidos) if cartas else "Nenhuma")
        largura_cartas = cv2.getTextSize(texto_cartas, font, 0.75, 2)[0][0]
        escala_cartas = min(0.75, 0.75 * largura_texto / max(1, largura_cartas))

        cv2.putText(img, nome_jogador, (x, 42), font, 0.85,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, texto_cartas, (x, 78), font, escala_cartas,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"Soma: {somas[indice]}", (x, 112), font, 0.75,
                    (255, 255, 255), 2, cv2.LINE_AA)

    cor_resultado = (0, 255, 0) if resultado.startswith("Vencedor") else (0, 200, 255)
    largura_resultado = cv2.getTextSize(resultado, font, 0.8, 2)[0][0]
    x_resultado = max(12, (largura - largura_resultado) // 2)
    cv2.putText(img, resultado, (x_resultado, 145), font, 0.8,
                cor_resultado, 2, cv2.LINE_AA)

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

    img_masked = cv2.bitwise_and(img, img, mask=area)
    return cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)

def detectar_cartas(frame, kp_img, des_img, template_data, reverse_order):
    caixas_globais = []

    for template in (reversed(template_data) if reverse_order else template_data):
        encontrar_multiplas_instancias(
            template['kp'], template['des'],
            kp_img, des_img,
            template['img'], frame,
            template_name=template['name'],
            min_matches=min_matches_para(len(template['kp'])),
            ratio=0.75,
            ransac_thresh=5.5,
            caixas_globais=caixas_globais,
        )

    return caixas_globais

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

    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0
    output_path = "jogo21_anotado.mp4"
    video_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (largura, altura),
    )
    if not video_writer.isOpened():
        logger.error("Não foi possível criar o vídeo de saída: %s", output_path)
        cap.release()
        return
    
    templates_dir = 'templates/'
    sift = cv2.SIFT_create()

    anterior = None
    rodada = 0
    template_data = carregarTemplates(sift, templates_dir)
    img_display = None
    caixas_globais = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if cena_nova(anterior, frame):
            rodada += 1
            img_trabalho = process_frame(frame, sift, template_data)
            kp_img, des_img = sift.detectAndCompute(img_trabalho, None)

            if rodada in (7, 9):
                caixas_globais = detectar_cartas(
                    frame, kp_img, des_img, template_data, reverse_order=True)
            else:
                caixas_globais = detectar_cartas(
                    frame, kp_img, des_img, template_data, reverse_order=False)

        img_display = frame.copy()
        desenhar_placar_no_frame(img_display, caixas_globais)
        video_writer.write(img_display)
            
        anterior = frame

        if img_display is not None:
            cv2.imshow('Jogo', cv2.resize(img_display, (1280, 720)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    logger.info("Vídeo anotado salvo em: %s", output_path)

if __name__ == '__main__':
    main()