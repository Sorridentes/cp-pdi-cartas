import cv2
import numpy as np

def carregarTemplates(sift_detector, templates_directory):
    template_data = []
    print(f"Procurando templates em: {templates_directory}")

    if not os.path.exists(templates_directory):
        print(f"Erro: Diretório de templates '{templates_directory}' não encontrado.")
        return template_data

    for filename in os.listdir(templates_directory):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            template_path = os.path.join(templates_directory, filename)
            template_name = os.path.splitext(filename)[0] # Nome da carta sem extensão
            template_img = cv2.imread(template_path, 0) # Carregar como escala de cinza

            if template_img is None:
                print(f"Não foi possível carregar o template: {template_path}")
                continue

            kp_template, des_template = sift_detector.detectAndCompute(template_img, None)

            if des_template is not None:
                template_data.append({
                    'name': template_name,
                    'img': template_img,
                    'kp': kp_template,
                    'des': des_template
                })
                print(f"Template '{template_name}' carregado com {len(kp_template)} keypoints.")
            else:
                print(f"Não foi possível extrair descritores para o template: {template_name}")

    print(f"Total de {len(template_data)} templates carregados e processados.")
    return template_data

def encontrar_multiplas_instancias(kp_template, des_template, kp_img, des_img,
                                   template_img, img_scene_final,
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

    print(f"  [{template_name}] knn={len(knn)} | mutuos={len(mutuos)} | "
          f"bons(crosscheck+ratio)={len(bons)} | min={min_matches}")

    if len(bons) < min_matches:
        return 0

    src_all = np.float32([kp_template[m.queryIdx].pt for m in bons])
    dst_all = np.float32([kp_img[m.trainIdx].pt for m in bons])

    indices = list(range(len(bons)))
    caixas_globais = caixas_globais if caixas_globais is not None else []
    n_inst = 0
    primeira_inst = True
    inliers_da_primeira = None
    PALETA = [(0,255,0),(0,0,255),(255,0,0),(0,255,255),(255,0,255),(255,255,0)]

    while len(indices) >= min_matches:
        src_pts = src_all[indices].reshape(-1,1,2)
        dst_pts = dst_all[indices].reshape(-1,1,2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        if H is None or mask is None:
            print(f"  [{template_name}] H=None, para")
            break

        inliers = mask.ravel().astype(bool)
        n_in = int(inliers.sum())
        if n_in < min_matches:
            break
        
        if primeira_inst:
          if n_in < min_matches:           # exige primeira detecção forte
              break
          inliers_da_primeira = n_in
          primeira_inst = False
        else:
          if n_in < 0.6 * inliers_da_primeira:
              # provavelmente fantasma (invertido ou lixo)
              # remove e tenta de novo? ou break? → break é mais seguro
              break

        ok = homografia_plausivel(H, template_img, img_scene_final.shape, 
                                  debug=True)
        print(f"  [{template_name}] geom_ok={ok}")
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
            if iou(box, b_old) > 0.3:
                sobreposto = True
                break

        if sobreposto:
            # remove esses inliers e continua procurando outra carta igual de verdade
            indices = [indices[i] for i in range(len(indices)) if not inliers[i]]
            continue

        caixas_globais.append((box, template_name, n_in))
        cor = PALETA[n_inst % len(PALETA)]
        desenhaContorno_com_H(H, template_img, img_scene_final, cor,
                              label=f"{template_name}#{n_inst+1}")
        n_inst += 1

        indices = [indices[i] for i in range(len(indices)) if not inliers[i]]

    return n_inst

def desenhaContorno_com_H(H, template_img, img_scene_final, cor, label=""):
    """Desenha o contorno de UMA instância usando a homografia já calculada."""
    if H is None:
        return
    h, w = template_img.shape[:2]
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)
    cv2.polylines(img_scene_final, [np.int32(dst)], True, cor, 3)

    # desenha o nome da carta no centro
    if label:
        cx = int(dst[:, 0, 0].mean())
        cy = int(dst[:, 0, 1].mean())
        cv2.putText(img_scene_final, label, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)

def homografia_plausivel(H, template_img, scene_shape, debug=False):
    h, w = template_img.shape[:2]

    # Ordem: TL, TR, BR, BL  (consistente com os nomes)
    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    # dst[0]=TL  dst[1]=TR  dst[2]=BR  dst[3]=BL

    if not cv2.isContourConvex(dst.astype(np.int32)):
        if debug: print("  geom: não convexo")
        return False

    area = abs(cv2.contourArea(dst.astype(np.float32)))
    ratio_area = area / (w * h)
    # if not (0.3 < ratio_area < 2.5):
    #     if debug: print(f"  geom: area_ratio={ratio_area:.2f}")
    #     return False

    def side(p, q): return np.linalg.norm(q - p)

    # Arestas horizontais = topo (TL→TR) e base (BL→BR)
    w_det = (side(dst[0], dst[1]) + side(dst[3], dst[2])) / 2
    # Arestas verticais = esquerda (TL→BL) e direita (TR→BR)
    h_det = (side(dst[0], dst[3]) + side(dst[1], dst[2])) / 2

    if h_det <= 1:
        if debug: print("  geom: h_det<=1")
        return False

    ar = (w_det / h_det) / (w / h)
    if not (0.5 < ar < 2.0):
        if debug: print(f"  geom: ar={ar:.2f} (w_det={w_det:.0f}, h_det={h_det:.0f})")
        return False

    ih, iw = scene_shape[:2]
    if dst[:, 0].min() < -50 or dst[:, 0].max() > iw + 50:
        if debug: print("  geom: fora horizontalmente")
        return False
    if dst[:, 1].min() < -50 or dst[:, 1].max() > ih + 50:
        if debug: print("  geom: fora verticalmente")
        return False

    if debug: print(f"  geom OK: ar={ar:.2f}, area_ratio={ratio_area:.2f}")
    return True

def iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter + 1e-6)

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
    min_y_regions = float('inf')
    max_y_regions = 0
    min_x_regions = float('inf')
    max_x_regions = 0

    if regioes: # Ensure regioes is not empty
        for c in regioes:
            x, y, w, h = cv2.boundingRect(c)
            min_y_regions = min(min_y_regions, y)
            max_y_regions = max(max_y_regions, y + h)
            min_x_regions = min(min_x_regions, x)
            max_x_regions = max(max_x_regions, x + w)
    else:
        H = 0
        min_y_regions = 0
        max_y_regions = img.shape[0] # Fallback to full image height if no regions
        min_x_regions = 0
        max_x_regions = img.shape[1] # Fallback to full image width if no regions
        print("No regions found to calculate H.")

    area_BGR = cv2.cvtColor(area, cv2.COLOR_GRAY2BGR)
    cropped_img = img.copy()
    cropped_img = cv2.bitwise_and(cropped_img, area_BGR)
    cropped_img = cropped_img[min_y_regions:max_y_regions, min_x_regions:max_x_regions]
    return cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

def detectar_cartas(frame, kp_img, des_img, template_data):
    img_scene_final = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    caixas_globais = []   # uma lista só, passada para todos os templates

    for template in template_data:
        n = encontrar_multiplas_instancias(
            template['kp'], template['des'],
            kp_img, des_img,
            template['img'], img_scene_final,
            template_name=template['name'],
            min_matches=8, 
            ratio=0.75, 
            ransac_thresh=5.5,
            caixas_globais=caixas_globais,
        )

        if n > 0:
            print(f"Carta '{template['name']}': {n} instância(s) encontrada(s).")
        else:
            print(f"Carta '{template['name']}': nenhuma instância encontrada.")

def main():
    cap = cv2.VideoCapture("jogo21.mp4") ## fonte de vídeo (0 para webcam padrão) ou pode ser um arquivo de vídeo, por exemplo: 'video.mp4', lembra de ajustar o path
    if not cap.isOpened():
        print("Erro: Não foi possível acessar a webcam")
        return
    templates_dir = 'templates/'
    # Inicializar SIFT
    sift = cv2.SIFT_create()

    template_data = carregarTemplates(sift, templates_dir)

    # Calcular keypoints e descritores para a imagem de cena (img) uma única vez
    kp_img, des_img = sift.detectAndCompute(img, None)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível capturar o frame")
            break
        # frame = cv2.flip(frame, 1)  # espelha o frame horizontalmente
        # frame = cv2.resize(frame, (640, 480))  # redimensiona o frame para 640x480
        
        # Aqui você pode adicionar processamento de imagem
        
        
        
        
        
        
        
        
        
        # aqui vc exibe o frame processado
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # Pressione 'q' para sair
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
