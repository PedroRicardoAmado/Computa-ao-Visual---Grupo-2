#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "opencv-python>=5.0,<6",
#   "rich",
# ]
# ///
"""A cena reage ao movimento: a aplicação do projeto de grupo de Computação Visual 2026/27.

    uv run src/app.py [pasta dos assets] [indice da camara, ou o endereço do DroidCam]

Parte do programa com que acaba a PL06 (secções 1 a 4 e o desafio 6): as zonas, as animações, os
cursores e o config.json da PL05; um evento por zona, com um vídeo e uma fotografia do início; o
registo.json; e a consola do rich. Para os 17 valores, acrescenta o que os desafios da PL05 deixaram
para explorar:

- a subtração de fundo com MOG2 ou KNN (desafio 1 da PL05 e desafio 2 da PL04), com a diferença de
  fotogramas ainda à mão, na tecla d;
- reações de tipos diferentes por zona (secção 5 e desafio 2 da PL05), escolhidas no config.json:
  normal (no meio da zona), movimento (onde a pessoa se mexeu), sentido (a animação corre para trás
  quando o movimento vai para a esquerda) e tamanho (o objeto cresce com o movimento na zona).

Teclas: d deteção · f aprender o fundo outra vez · v máscara · g gravar config.json · q sair
"""
import json
import pathlib
import sys
from datetime import datetime

import cv2 as cv
import numpy as np
from rich.console import Console

raiz = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "assets")
fonte = sys.argv[2] if len(sys.argv) > 2 else "0"
fonte = int(fonte) if fonte.isdigit() else fonte     # 0, 1, ... ou "http://IP:4747/video"

CONFIG = pathlib.Path("config.json")
EVIDENCIAS = pathlib.Path("evidencias")
REGISTO = pathlib.Path("registo.json")
DETECOES = ("diferenca", "mog2", "knn")
TIPOS = ("normal", "movimento", "sentido", "tamanho")
OMISSAO = {"limiar": 25, "percentagem": 2, "paragem": 15, "detecao": "mog2",
           "zonas": [[0, 0, 213, 480], [213, 0, 213, 480],
                     [426, 0, 214, 480]],
           "tipos": ["movimento", "sentido", "tamanho"]}
LADO = 160
AREA_MINIMA = 500              # só para as caixas
APRENDIZAGEM = 0.001           # MOG2 e KNN aprendem devagar: quem pára continua marcado
AQUECIMENTO = 60               # fotogramas, no arranque, a aprender o fundo sem acender zonas
ELIPSE = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
VERDE, AMARELO, VERMELHO = (0, 255, 0), (0, 255, 255), (0, 0, 255)
JANELA, JANELA_MASCARA = "esplanada", "mascara"

consola = Console(log_path=False)                # cada mensagem com a hora à esquerda


def carrega():
    """A configuração do ficheiro, com os valores por omissão para o que lá não estiver."""
    valores = dict(OMISSAO)
    if CONFIG.exists():
        try:
            valores.update(json.loads(CONFIG.read_text(encoding="utf-8")))
        except json.JSONDecodeError as e:
            sys.exit(f"{CONFIG} não é um JSON válido ({e}). Apaguem-no para "
                     "recomeçar com os valores por omissão.")
    for n, roi in enumerate(valores["zonas"], start=1):
        if len(roi) != 4:
            sys.exit(f"{CONFIG}: a zona {n} tem de ser [x, y, largura, altura], "
                     f"e está {roi}.")
    if valores["detecao"] not in DETECOES:
        sys.exit(f"{CONFIG}: a deteção tem de ser uma de {DETECOES}, e está "
                 f"{valores['detecao']!r}.")
    if len(valores["tipos"]) != len(valores["zonas"]):
        sys.exit(f"{CONFIG}: há {len(valores['zonas'])} zonas e {len(valores['tipos'])} "
                 "tipos de reação: tem de haver um tipo por zona.")
    for n, tipo in enumerate(valores["tipos"], start=1):
        if tipo not in TIPOS:
            sys.exit(f"{CONFIG}: o tipo da zona {n} tem de ser um de {TIPOS}, e está {tipo!r}.")
    return valores


def le_sequencia(pasta):
    fotogramas = []
    for ficheiro in sorted(pasta.glob("*.png")):
        objeto = cv.imread(str(ficheiro), cv.IMREAD_UNCHANGED)
        if objeto is None:
            sys.exit(f"Não consegui ler {ficheiro.name}. O caminho está certo?")
        if objeto.ndim != 3 or objeto.shape[2] != 4:
            sys.exit(f"{ficheiro.name} não tem canal alfa (transparência). No "
                     "Blender: Render Properties › Film › Transparent, e no "
                     "Output, PNG com RGBA.")
        fotogramas.append(cv.resize(objeto, (LADO, LADO)))
    if not fotogramas:
        sys.exit(f"Não há PNG em {pasta}. A pasta desta zona está vazia?")
    return fotogramas


def confere(zonas, largura, altura):
    """Pára com uma mensagem se alguma zona sair da imagem."""
    for n, (x, y, w, h) in enumerate(zonas, start=1):
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > largura or y + h > altura:
            sys.exit(f"A zona {n} {(x, y, w, h)} sai da imagem de {largura} x {altura}. "
                     f"Corrijam as zonas no {CONFIG}.")


def compoe(fundo, objeto, x, y):
    """Mistura por alfa do objeto (BGRA) no fundo, com o canto superior esquerdo em (x, y)."""
    h, w = objeto.shape[:2]
    x = max(0, min(x, fundo.shape[1] - w))       # não deixar sair da imagem: encosta
    y = max(0, min(y, fundo.shape[0] - h))
    fatia = fundo[y:y + h, x:x + w]
    if fatia.shape[:2] != (h, w):                # objeto maior do que a própria imagem
        return
    alfa = objeto[:, :, 3] / 255.0
    alfa = alfa[:, :, None]
    fundo[y:y + h, x:x + w] = (alfa * objeto[:, :, :3] + (1 - alfa) * fatia).astype("uint8")


def reage(fotograma, objeto, zona, tipo, onde, fracao):
    """Compõe o objeto da zona conforme o tipo de reação dela."""
    x, y, w, h = zona
    if tipo == "tamanho":                        # cresce com a parte da zona que está a branco
        escala = min(2.0, max(0.5, fracao / 0.10))
        lado = max(40, int(LADO * escala))
        objeto = cv.resize(objeto, (lado, lado))
    lado = objeto.shape[0]
    if tipo == "movimento" and onde is not None:  # onde a pessoa se mexeu
        compoe(fotograma, objeto, onde[0] - lado // 2, onde[1] - lado // 2)
    else:                                         # no meio da zona
        compoe(fotograma, objeto, x + (w - lado) // 2, y + (h - lado) // 2)


def cria_subtrator(detecao):
    """O modelo do fundo do MOG2 ou do KNN, como nos desafios; None na diferença de fotogramas."""
    if detecao == "mog2":
        return cv.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True)
    if detecao == "knn":
        return cv.createBackgroundSubtractorKNN(
            history=500, dist2Threshold=400.0, detectShadows=True)
    return None


def toca(caixa, zona):
    """True se a caixa e a zona se sobrepõem, nem que seja numa ponta."""
    x, y, w, h = caixa
    zx, zy, zw, zh = zona
    return x < zx + zw and zx < x + w and y < zy + zh and zy < y + h


def carimba(imagem):
    """Uma cópia da imagem com a data e a hora escritas no canto."""
    copia = imagem.copy()
    agora = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    for cor, grossura in (((0, 0, 0), 4), ((255, 255, 255), 2)):
        cv.putText(copia, agora, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                   cor, grossura)
    return copia


def comeca(z, fotograma, caixas, fps):
    """Um evento novo na zona z: a hora de início, as caixas nela, o vídeo e a fotografia."""
    inicio = datetime.now()
    evento = {"inicio": inicio, "caixas": caixas}
    pasta = EVIDENCIAS / f"{inicio:%Y-%m-%d}"
    pasta.mkdir(parents=True, exist_ok=True)
    nome = f"zona{z + 1}_{inicio:%Y-%m-%d_%H-%M-%S}"
    ficheiro = pasta / f"{nome}.mp4"
    n = 2
    while ficheiro.exists():                     # dois eventos da zona no mesmo segundo
        ficheiro = pasta / f"{nome}_{n}.mp4"
        n += 1
    altura, largura = fotograma.shape[:2]
    video = cv.VideoWriter(str(ficheiro), cv.VideoWriter_fourcc(*"mp4v"),
                           fps, (largura, altura))
    if not video.isOpened():
        consola.log(f"[red]zona {z + 1}: não consegui gravar {ficheiro}[/]")
        video, ficheiro = None, None
    evento["video"], evento["ficheiro"] = video, ficheiro
    if ficheiro is not None:                     # a fotografia do início
        foto = fotograma.copy()
        for x, y, w, h in caixas:
            cv.rectangle(foto, (x, y), (x + w, y + h), AMARELO, 2)
        cv.imwrite(str(ficheiro.with_suffix(".png")), foto)
    consola.log(f"[green]zona {z + 1}[/]: começou; a gravar {ficheiro}")
    return evento


def le_registo():
    """Os eventos já registados; uma lista vazia se ainda não há ficheiro."""
    if not REGISTO.exists():
        return []
    try:
        registo = json.loads(REGISTO.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"{REGISTO} não é um JSON válido ({e}). Corrijam-no ou "
                 "mudem-lhe o nome, para o programa começar um registo novo.")
    if not isinstance(registo, list):
        sys.exit(f"{REGISTO} tem de ser uma lista de eventos, [ ... ].")
    return registo


def acaba(z, evento):
    """Fecha o evento da zona z: o vídeo, a consola e uma entrada no registo.json."""
    duracao = (datetime.now() - evento["inicio"]).total_seconds()
    if evento["video"] is not None:
        evento["video"].release()
    consola.log(f"[yellow]zona {z + 1}[/]: acabou, {duracao:.1f} s")
    registo = le_registo()
    ficheiro = evento["ficheiro"]
    registo.append({
        "zona": z + 1,
        "inicio": evento["inicio"].isoformat(timespec="seconds"),
        "duracao_s": round(duracao, 1),
        "ficheiro": ficheiro.as_posix() if ficheiro else None,
        "caixas": [list(c) for c in evento["caixas"]],
        "objeto": pastas[z].name,
    })
    REGISTO.write_text(json.dumps(registo, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")


valores = carrega()
zonas = [tuple(int(v) for v in roi) for roi in valores["zonas"]]
tipos = valores["tipos"]
pastas = sorted(p for p in raiz.glob("zona*") if p.is_dir())
if not pastas:
    sys.exit(f"Não há pastas de zonas em {raiz} (esperava assets/zona1_..., "
             "assets/zona2_..., uma por zona).")
if len(pastas) != len(zonas):
    sys.exit(f"Há {len(zonas)} zonas no {CONFIG} e {len(pastas)} pastas em "
             f"{raiz}: tem de haver uma pasta por zona.")
sequencias = [le_sequencia(p) for p in pastas]
consola.log("configuração:", valores)
consola.log(f"{REGISTO}: {len(le_registo())} evento(s) de antes")

cv.namedWindow(JANELA)
cv.createTrackbar("limiar", JANELA, valores["limiar"], 100, lambda v: None)
cv.createTrackbar("zona %", JANELA, valores["percentagem"], 50, lambda v: None)
cv.createTrackbar("paragem", JANELA, valores["paragem"], 60, lambda v: None)

camara = cv.VideoCapture(fonte)
if not camara.isOpened():
    sys.exit(f"Não consegui abrir a câmara {fonte}.")
camara.set(cv.CAP_PROP_FRAME_WIDTH, 640)           # as zonas estão em píxeis de 640 x 480
camara.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
fps = camara.get(cv.CAP_PROP_FPS) or 30            # há câmaras que dizem 0

i = [0] * len(zonas)                               # o fotograma da animação de cada zona
quietos = [valores["paragem"]] * len(zonas)        # fotogramas sem movimento, por zona
eventos = [None] * len(zonas)                      # o evento aberto em cada zona
onde = [None] * len(zonas)                         # o centro do movimento em cada zona
sentido = [1] * len(zonas)                         # +1 para a frente, -1 para trás
anterior = None
subtrator = cria_subtrator(valores["detecao"])
aquecer = AQUECIMENTO
ver_mascara = False
try:
    while True:
        ok, fotograma = camara.read()
        if not ok:
            break
        valores["limiar"] = cv.getTrackbarPos("limiar", JANELA)
        valores["percentagem"] = cv.getTrackbarPos("zona %", JANELA)
        valores["paragem"] = max(1, cv.getTrackbarPos("paragem", JANELA))
        altura, largura = fotograma.shape[:2]
        confere(zonas, largura, altura)
        cru = carimba(fotograma)                   # a da câmara, sem desenhos, com a hora

        # A máscara do movimento: diferença de fotogramas (PL04) ou subtração de fundo (MOG2, KNN).
        mascara = None
        if subtrator is None:
            suave = cv.GaussianBlur(cv.cvtColor(fotograma, cv.COLOR_BGR2GRAY), (21, 21), 0)
            if anterior is not None:
                _, limiar = cv.threshold(cv.absdiff(anterior, suave), valores["limiar"],
                                         255, cv.THRESH_BINARY)
                mascara = cv.dilate(limiar, None, iterations=2)
            anterior = suave
        else:
            taxa = -1 if aquecer > 0 else APRENDIZAGEM     # no arranque aprende depressa
            bruta = subtrator.apply(fotograma, learningRate=taxa)
            # as sombras saem a cinzento (127): só o branco conta como objeto
            _, so_objeto = cv.threshold(bruta, 200, 255, cv.THRESH_BINARY)
            so_objeto = cv.morphologyEx(so_objeto, cv.MORPH_OPEN, ELIPSE)
            mascara = cv.dilate(so_objeto, None, iterations=2)
        if aquecer > 0:                            # a aprender o fundo: nenhuma zona acende
            aquecer -= 1
            mascara = None

        acesas, caixas = set(), []
        fracao = [0.0] * len(zonas)                # a parte de cada zona a branco, de 0 a 1
        if mascara is not None:
            for z, (x, y, w, h) in enumerate(zonas):
                fatia = mascara[y:y + h, x:x + w]
                brancos = cv.countNonZero(fatia)
                fracao[z] = brancos / (w * h)
                if brancos > valores["percentagem"] / 100 * w * h:
                    acesas.add(z)
                    ys, xs = np.nonzero(fatia)     # onde estão os brancos da zona
                    cx, cy = x + int(xs.mean()), y + int(ys.mean())
                    if onde[z] is not None and abs(cx - onde[z][0]) > 3:
                        sentido[z] = 1 if cx > onde[z][0] else -1
                    onde[z] = (cx, cy)
            contornos, _ = cv.findContours(mascara, cv.RETR_EXTERNAL,
                                           cv.CHAIN_APPROX_SIMPLE)
            for contorno in contornos:
                if cv.contourArea(contorno) >= AREA_MINIMA:
                    caixas.append(cv.boundingRect(contorno))
            for x, y, w, h in caixas:
                cv.rectangle(fotograma, (x, y), (x + w, y + h), AMARELO, 2)

        for z, zona in enumerate(zonas):
            quietos[z] = 0 if z in acesas else quietos[z] + 1
            if z in acesas and eventos[z] is None:     # começa um evento
                dela = [c for c in caixas if toca(c, zona)]
                eventos[z] = comeca(z, cru, dela, fps)
            if eventos[z] is not None and eventos[z]["video"] is not None:
                eventos[z]["video"].write(cru)
            if eventos[z] is not None and quietos[z] >= valores["paragem"]:
                acaba(z, eventos[z])                   # acaba quando a zona sossega
                eventos[z] = None
                onde[z], sentido[z] = None, 1
            x, y, w, h = zona
            if quietos[z] < valores["paragem"]:        # esta zona está a animar
                objeto = sequencias[z][i[z] % len(sequencias[z])]
                reage(fotograma, objeto, zona, tipos[z], onde[z], fracao[z])
                i[z] += sentido[z] if tipos[z] == "sentido" else 1
            else:
                i[z] = 0                               # parada: volta ao princípio
            cv.rectangle(fotograma, (x, y), (x + w - 1, y + h - 1), VERDE,
                         4 if z in acesas else 1)
            cv.putText(fotograma, f"{z + 1} {tipos[z]}", (x + 8, y + 28),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, VERDE, 2)

        if any(e is not None for e in eventos):        # o aviso de que grava
            cv.circle(fotograma, (largura - 70, 25), 8, VERMELHO, -1)
            cv.putText(fotograma, "REC", (largura - 55, 33),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, VERMELHO, 2)
        estado = "a aprender o fundo" if aquecer > 0 else valores["detecao"]
        cv.putText(fotograma, f"{estado} | d detecao  f fundo  v mascara  g gravar  q sair",
                   (10, altura - 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, VERDE, 1)
        cv.imshow(JANELA, fotograma)
        if ver_mascara and mascara is not None:
            cv.imshow(JANELA_MASCARA, mascara)

        tecla = cv.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("g"):
            CONFIG.write_text(json.dumps(valores, indent=2) + "\n", encoding="utf-8")
            consola.log("gravado em", CONFIG, valores)
        if tecla in (ord("d"), ord("f")):
            if tecla == ord("d"):                      # diferença -> MOG2 -> KNN -> diferença
                n = (DETECOES.index(valores["detecao"]) + 1) % len(DETECOES)
                valores["detecao"] = DETECOES[n]
            subtrator = cria_subtrator(valores["detecao"])   # o fundo aprende-se do zero
            anterior, aquecer = None, AQUECIMENTO
            consola.log(f"[cyan]deteção: {valores['detecao']}[/], a aprender o fundo")
        if tecla == ord("v"):
            ver_mascara = not ver_mascara
            if not ver_mascara:
                try:
                    cv.destroyWindow(JANELA_MASCARA)
                except cv.error:                       # a janela ainda não tinha aberto
                    pass
finally:
    for z, evento in enumerate(eventos):           # quem sai com q, com Ctrl+C ou num erro
        if evento is not None:                     # a meio de um evento, não o perde
            acaba(z, evento)
    camara.release()
    cv.destroyAllWindows()
