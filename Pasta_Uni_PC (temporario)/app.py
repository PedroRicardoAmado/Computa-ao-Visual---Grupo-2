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
  fotogramas ainda à mão, na tecla d; o cursor do limiar afina os três modos;
- reações de tipos diferentes por zona (secção 5 e desafio 2 da PL05), escolhidas no config.json:
  normal (no meio da zona), movimento (onde a pessoa se mexeu), sentido (a animação corre para trás
  quando o movimento vai para a esquerda) e tamanho (o objeto cresce com o movimento na zona);
- uma interface mais legível: uma cor por zona, as zonas acesas pintadas por cima da câmara
  (cv.addWeighted, dos slides da S05), etiquetas, uma barra de estado, o REC a piscar e a máscara
  em miniatura, no canto (tecla v). A evidência gravada continua a ser a imagem limpa da câmara.

O ficheiro está por esta ordem: a configuração; as funções, por tema (config.json e PNG, deteção,
desenho e reações, eventos e registo); o arranque; e o ciclo principal, com os passos numerados.

Teclas: d deteção · f fundo outra vez · v máscara no canto · p cabeça e mãos · g gravar config.json · q sair
"""
import json
import pathlib
import sys
from datetime import datetime

import cv2 as cv
import numpy as np
from rich.console import Console

# --- Configuração --------------------------------------------------------------------------------

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
LADO = 160                     # lado, em píxeis, a que se reduzem os PNG
AREA_MINIMA = 3000             # área mínima de uma caixa de movimento (píxeis)
JUNTAR = 80                    # manchas a menos de tantos píxeis ficam na mesma caixa
MAX_CAIXAS = 3                 # no máximo, as maiores caixas por fotograma
LIGACAO = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * JUNTAR + 1, 2 * JUNTAR + 1))
APRENDIZAGEM = 0.001           # MOG2 e KNN aprendem devagar: quem pára continua marcado
AQUECIMENTO = 60               # fotogramas, no arranque, a aprender o fundo sem acender zonas
LARGURA, ALTURA = 640, 480     # as zonas do config.json estão em píxeis de 640 x 480
FALHAS_MAX = 30                # fotogramas seguidos sem imagem antes de desistir
DESCARTAR = 10                 # fotogramas deitados fora ao abrir: a câmara acerta a exposição
ELIPSE = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
AMARELO, VERMELHO = (0, 255, 255), (0, 0, 255)
BRANCO, PRETO, CINZENTO = (255, 255, 255), (0, 0, 0), (190, 190, 190)
CORES = [(255, 191, 0), (113, 204, 46), (94, 197, 255),    # uma cor por zona, em BGR: azul,
         (180, 105, 255), (0, 215, 255)]                   # verde, laranja, rosa e dourado
MAGENTA, LARANJA = (255, 0, 255), (0, 140, 255)       # cabeça e mãos
PELE_MIN, PELE_MAX = (0, 133, 77), (255, 173, 127)    # a pele em YCrCb: (Y, Cr, Cb)
AREA_PELE = 1200               # área mínima de uma mancha de pele (píxeis)
MAOS_MAX = 2                   # no máximo duas mãos
JANELA = "esplanada"

consola = Console(log_path=False)                # cada mensagem com a hora à esquerda


# --- O config.json e as sequências PNG -----------------------------------------------------------

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
    """Os PNG da pasta, por ordem, com o canal alfa e reduzidos a LADO x LADO."""
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


# --- A câmara ------------------------------------------------------------------------------------

def abre_camara(fonte):
    """Abre a câmara e confirma que chega imagem de verdade; None se não conseguir.

    Num índice (0, 1, ...) experimenta os motores de vídeo do sistema (no Windows, o DirectShow
    abre muito mais depressa e com menos falhas do que o MSMF) e, se o índice pedido não der,
    os vizinhos de 0 a 3. Num endereço (DroidCam, IP) usa o motor por omissão.
    """
    if isinstance(fonte, int):
        if sys.platform.startswith("win"):
            motores = [cv.CAP_DSHOW, cv.CAP_MSMF, cv.CAP_ANY]
        elif sys.platform == "darwin":
            motores = [cv.CAP_AVFOUNDATION, cv.CAP_ANY]
        else:
            motores = [cv.CAP_V4L2, cv.CAP_ANY]
        candidatos = [fonte] + [n for n in range(4) if n != fonte]
    else:
        motores, candidatos = [cv.CAP_ANY], [fonte]
    for candidato in candidatos:
        for motor in motores:
            camara = cv.VideoCapture(candidato, motor)
            if not camara.isOpened():
                camara.release()
                continue
            camara.set(cv.CAP_PROP_FRAME_WIDTH, LARGURA)
            camara.set(cv.CAP_PROP_FRAME_HEIGHT, ALTURA)
            camara.set(cv.CAP_PROP_BUFFERSIZE, 1)          # sem atraso acumulado
            recebeu = False
            for _ in range(50):                            # a câmara pode demorar a enviar
                ok, imagem = camara.read()
                if ok and imagem is not None and imagem.size:
                    recebeu = True
                    break
                cv.waitKey(50)
            if not recebeu:
                camara.release()
                continue
            for _ in range(DESCARTAR):                     # as primeiras imagens vêm escuras
                camara.read()
            consola.log(f"câmara {candidato!r} aberta (motor {motor})")
            return camara
    return None


def ajusta(fotograma):
    """Garante LARGURA x ALTURA (4:3): corta o excesso do lado e reduz, sem esticar a imagem.

    Assim as zonas do config.json servem a qualquer câmara, mesmo a que ignora o pedido de 640 x 480.
    """
    altura, largura = fotograma.shape[:2]
    if (largura, altura) == (LARGURA, ALTURA):
        return fotograma
    alvo = LARGURA / ALTURA
    if largura / altura > alvo:                    # mais larga do que 4:3: corta dos lados
        nova = int(altura * alvo)
        x = (largura - nova) // 2
        fotograma = fotograma[:, x:x + nova]
    elif largura / altura < alvo:                  # mais alta do que 4:3: corta em cima e em baixo
        nova = int(largura / alvo)
        y = (altura - nova) // 2
        fotograma = fotograma[y:y + nova, :]
    return cv.resize(fotograma, (LARGURA, ALTURA))


# --- A deteção do movimento ----------------------------------------------------------------------

def cria_subtrator(detecao):
    """O modelo do fundo do MOG2 ou do KNN, como nos desafios; None na diferença de fotogramas."""
    if detecao == "mog2":
        return cv.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True)
    if detecao == "knn":
        return cv.createBackgroundSubtractorKNN(
            history=500, dist2Threshold=400.0, detectShadows=True)
    return None


def mascara_diferenca(fotograma, anterior, limiar):
    """Diferença de fotogramas (PL04): a máscara e o fotograma suave, que fica como anterior.

    No primeiro fotograma ainda não há anterior, e a máscara é None.
    """
    suave = cv.GaussianBlur(cv.cvtColor(fotograma, cv.COLOR_BGR2GRAY), (21, 21), 0)
    if anterior is None:
        return None, suave
    _, binaria = cv.threshold(cv.absdiff(anterior, suave), limiar, 255, cv.THRESH_BINARY)
    return cv.dilate(binaria, None, iterations=2), suave


def mascara_fundo(subtrator, fotograma, aprendizagem, limiar):
    """Subtração de fundo com MOG2 ou KNN (desafio 1 da PL05): a branco, o que não é fundo.

    O limiar dos cursores também afina estes dois modos: quanto maior, menos sensível.
    """
    if isinstance(subtrator, cv.BackgroundSubtractorMOG2):
        subtrator.setVarThreshold(max(1, limiar))
    else:
        subtrator.setDist2Threshold(max(1, limiar) * 16.0)   # 25 x 16 = 400, o dos desafios
    bruta = subtrator.apply(fotograma, learningRate=aprendizagem)
    _, so_objeto = cv.threshold(bruta, 200, 255, cv.THRESH_BINARY)   # as sombras (127) não contam
    so_objeto = cv.morphologyEx(so_objeto, cv.MORPH_OPEN, ELIPSE)     # apaga os pontos soltos
    return cv.dilate(so_objeto, None, iterations=2)                   # junta as manchas


def zonas_com_movimento(mascara, zonas, percentagem):
    """As zonas acesas, a parte de cada zona a branco (0 a 1) e o centro dos brancos das acesas."""
    acesas = set()
    fracao = [0.0] * len(zonas)
    centros = [None] * len(zonas)
    for z, (x, y, w, h) in enumerate(zonas):
        fatia = mascara[y:y + h, x:x + w]
        brancos = cv.countNonZero(fatia)
        fracao[z] = brancos / (w * h)
        if brancos > percentagem / 100 * w * h:
            acesas.add(z)
            ys, xs = np.nonzero(fatia)               # onde estão os brancos da zona
            centros[z] = (x + int(xs.mean()), y + int(ys.mean()))
    return acesas, fracao, centros


def funde_caixas(caixas, folga=0):
    """Junta as caixas que se sobrepõem (ou ficam a menos de folga píxeis) numa só, até não haver mais."""
    caixas = [list(c) for c in caixas]
    mudou = True
    while mudou:
        mudou = False
        for a in range(len(caixas)):
            for b in range(a + 1, len(caixas)):
                ax, ay, aw, ah = caixas[a]
                bx, by, bw, bh = caixas[b]
                if (ax - folga < bx + bw and bx - folga < ax + aw
                        and ay - folga < by + bh and by - folga < ay + ah):
                    x, y = min(ax, bx), min(ay, by)
                    caixas[a] = [x, y, max(ax + aw, bx + bw) - x, max(ay + ah, by + bh) - y]
                    del caixas[b]
                    mudou = True
                    break
            if mudou:
                break
    return [tuple(c) for c in caixas]


def caixas_do_movimento(mascara):
    """Uma caixa (x, y, largura, altura) por região com movimento, e não uma por mancha.

    Uma pessoa a mexer-se dá muitas manchas pequenas (mãos, cabeça, tronco). O fecho morfológico
    liga as que estão perto, e funde_caixas junta as caixas que ainda se tocam.
    """
    ligada = cv.morphologyEx(mascara, cv.MORPH_CLOSE, LIGACAO)
    contornos, _ = cv.findContours(ligada, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    caixas = [cv.boundingRect(c) for c in contornos if cv.contourArea(c) >= AREA_MINIMA]
    caixas = funde_caixas(caixas, JUNTAR // 2)
    caixas.sort(key=lambda c: c[2] * c[3], reverse=True)     # as maiores primeiro
    return caixas[:MAX_CAIXAS]


def caixas_de_pele(fotograma):
    """As caixas das manchas de cor de pele (visão clássica: cor + morfologia + contornos).

    Usa YCrCb em vez de BGR porque separa a cor (Cr, Cb) da luz (Y), e a pele cai quase sempre
    no mesmo intervalo de Cr e Cb, em tons claros e escuros.
    """
    ycrcb = cv.cvtColor(cv.GaussianBlur(fotograma, (7, 7), 0), cv.COLOR_BGR2YCrCb)
    pele = cv.inRange(ycrcb, PELE_MIN, PELE_MAX)
    pele = cv.morphologyEx(pele, cv.MORPH_OPEN, ELIPSE)               # apaga pontos soltos
    pele = cv.morphologyEx(pele, cv.MORPH_CLOSE, ELIPSE, iterations=2)  # fecha buracos
    contornos, _ = cv.findContours(pele, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return [cv.boundingRect(c) for c in contornos if cv.contourArea(c) >= AREA_PELE]


def cabeca_e_maos(caixas):
    """Separa as manchas de pele em cabeça (uma) e mãos (até MAOS_MAX).

    Regra: a cabeça é a maior mancha com proporções de cara (largura entre 0,5 e 1,3 vezes a
    altura); as restantes, da maior para a menor, são mãos. É uma heurística, não reconhece
    nada: braços descobertos ou objetos cor de pele podem enganá-la.
    """
    maiores = sorted(caixas, key=lambda c: c[2] * c[3], reverse=True)
    cabeca = next((c for c in maiores if 0.5 <= c[2] / c[3] <= 1.3), None)
    maos = [c for c in maiores if c != cabeca and not (cabeca and toca(c, cabeca))]
    return cabeca, maos[:MAOS_MAX]


# --- O desenho e as reações ----------------------------------------------------------------------

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


def sombreia(imagem, x, y, w, h, cor, opacidade):
    """Pinta um retângulo da imagem com a cor, meio transparente (cv.addWeighted)."""
    fatia = imagem[y:y + h, x:x + w]
    imagem[y:y + h, x:x + w] = cv.addWeighted(np.full_like(fatia, cor), opacidade,
                                              fatia, 1 - opacidade, 0)


def etiqueta(imagem, texto, x, y, cor):
    """O texto a preto num retângulo cheio da cor, para se ler em qualquer fundo."""
    (w, h), base = cv.getTextSize(texto, cv.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv.rectangle(imagem, (x, y), (x + w + 12, y + h + base + 10), cor, -1)
    cv.putText(imagem, texto, (x + 6, y + h + 5), cv.FONT_HERSHEY_SIMPLEX, 0.55,
               PRETO, 1, cv.LINE_AA)


def sombreia_zonas(imagem, zonas, acesas):
    """As zonas acesas ficam pintadas com a sua cor, por baixo das caixas e dos objetos."""
    for z in acesas:
        x, y, w, h = zonas[z]
        sombreia(imagem, x, y, w, h, CORES[z % len(CORES)], 0.18)


def desenha_zonas(imagem, zonas, tipos, acesas):
    """As molduras das zonas, cada uma da sua cor e mais grossas se acesas, e as etiquetas."""
    for z, (x, y, w, h) in enumerate(zonas):
        cor = CORES[z % len(CORES)]
        cv.rectangle(imagem, (x, y), (x + w - 1, y + h - 1), cor, 3 if z in acesas else 1)
        etiqueta(imagem, f"{z + 1}  {tipos[z]}", x + 6, y + 6, cor)


def miniatura(imagem, mascara):
    """A máscara do movimento em pequeno, no canto de baixo à direita (tecla v)."""
    altura, largura = imagem.shape[:2]
    w, h = largura // 4, altura // 4
    x, y = largura - w - 10, altura - h - 40
    imagem[y:y + h, x:x + w] = cv.cvtColor(cv.resize(mascara, (w, h)), cv.COLOR_GRAY2BGR)
    cv.rectangle(imagem, (x - 1, y - 1), (x + w, y + h), BRANCO, 1)


def desenha_estado(imagem, a_gravar, detecao, progresso):
    """A barra de baixo, com a deteção (ou o fundo a aprender) e as teclas, e o REC a piscar.

    a_gravar é o número de eventos abertos; progresso vai de 0 a 1 enquanto o fundo se aprende.
    """
    altura, largura = imagem.shape[:2]
    sombreia(imagem, 0, altura - 30, largura, 30, PRETO, 0.6)
    if progresso < 1:                              # uma barra que enche enquanto aprende
        cv.putText(imagem, "a aprender o fundo", (10, altura - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, BRANCO, 1, cv.LINE_AA)
        cv.rectangle(imagem, (170, altura - 20), (270, altura - 10), CINZENTO, 1)
        cv.rectangle(imagem, (170, altura - 20), (170 + int(100 * progresso), altura - 10),
                     BRANCO, -1)
    else:
        cv.putText(imagem, f"detecao: {detecao}", (10, altura - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, BRANCO, 1, cv.LINE_AA)
    cv.putText(imagem, "d detecao  f fundo  v mascara  p pele  g gravar  q sair",
               (largura - 410, altura - 10), cv.FONT_HERSHEY_SIMPLEX, 0.42, CINZENTO, 1,
               cv.LINE_AA)
    if a_gravar:
        x, y = largura - 104, 40
        cv.rectangle(imagem, (x, y), (largura - 10, y + 26), PRETO, -1)
        if datetime.now().microsecond < 500_000:   # a bola acende e apaga a cada meio segundo
            cv.circle(imagem, (x + 14, y + 13), 6, VERMELHO, -1, cv.LINE_AA)
        cv.putText(imagem, f"REC {a_gravar}", (x + 28, y + 19),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, BRANCO, 1, cv.LINE_AA)


# --- Os eventos, as evidências e o registo -------------------------------------------------------

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


# --- O arranque ----------------------------------------------------------------------------------

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

camara = abre_camara(fonte)
if camara is None:
    sys.exit(f"Não consegui abrir a câmara {fonte}. Está ligada e livre (outra aplicação, como o "
             "Zoom ou o Teams, pode estar a usá-la)? Tentem outro índice: "
             "uv run src/app.py assets 1")
fps = camara.get(cv.CAP_PROP_FPS)
if not 5 <= fps <= 60:                             # há câmaras que dizem 0 ou valores absurdos
    fps = 30

cv.namedWindow(JANELA)
cv.createTrackbar("limiar", JANELA, valores["limiar"], 100, lambda v: None)
cv.createTrackbar("zona %", JANELA, valores["percentagem"], 50, lambda v: None)
cv.createTrackbar("paragem", JANELA, valores["paragem"], 60, lambda v: None)

i = [0] * len(zonas)                               # o fotograma da animação de cada zona
quietos = [valores["paragem"]] * len(zonas)        # fotogramas sem movimento, por zona
eventos = [None] * len(zonas)                      # o evento aberto em cada zona
onde = [None] * len(zonas)                         # o centro do movimento em cada zona
sentido = [1] * len(zonas)                         # +1 para a frente, -1 para trás
anterior = None                                    # o fotograma anterior (diferença)
subtrator = cria_subtrator(valores["detecao"])     # o modelo do fundo (MOG2 ou KNN)
aquecer = AQUECIMENTO
ver_mascara = False
falhas = 0                                         # fotogramas seguidos sem imagem
ver_pele = True                                    # cabeça e mãos (tecla p)

# --- O ciclo principal ---------------------------------------------------------------------------

try:
    while True:
        # 1. A imagem da câmara e os valores dos cursores
        ok, fotograma = camara.read()
        if not ok or fotograma is None:            # uma falha solta não deve fechar a aplicação
            falhas += 1
            if falhas >= FALHAS_MAX:
                consola.log("[red]a câmara deixou de enviar imagem[/]")
                break
            cv.waitKey(10)
            continue
        falhas = 0
        fotograma = ajusta(fotograma)              # sempre 640 x 480, como as zonas
        valores["limiar"] = cv.getTrackbarPos("limiar", JANELA)
        valores["percentagem"] = cv.getTrackbarPos("zona %", JANELA)
        valores["paragem"] = max(1, cv.getTrackbarPos("paragem", JANELA))
        altura, largura = fotograma.shape[:2]
        confere(zonas, largura, altura)
        cru = carimba(fotograma)                   # a evidência: sem desenhos, com a hora

        # 2. A máscara do movimento
        if subtrator is None:
            mascara, anterior = mascara_diferenca(fotograma, anterior, valores["limiar"])
        else:
            taxa = -1 if aquecer > 0 else APRENDIZAGEM     # no arranque aprende depressa
            mascara = mascara_fundo(subtrator, fotograma, taxa, valores["limiar"])
        if aquecer > 0:                            # a aprender o fundo: nenhuma zona acende
            aquecer -= 1
            mascara = None

        # 3. As zonas acesas, as caixas e o sentido do movimento em cada zona
        acesas, caixas = set(), []
        fracao, centros = [0.0] * len(zonas), [None] * len(zonas)
        if mascara is not None:
            acesas, fracao, centros = zonas_com_movimento(mascara, zonas,
                                                          valores["percentagem"])
            caixas = caixas_do_movimento(mascara)
        for z in acesas:
            if onde[z] is not None and abs(centros[z][0] - onde[z][0]) > 3:
                sentido[z] = 1 if centros[z][0] > onde[z][0] else -1
            onde[z] = centros[z]

        cabeca, maos = cabeca_e_maos(caixas_de_pele(fotograma)) if ver_pele else (None, [])

        # 4. Os eventos: começam quando a zona acende e acabam quando ela sossega
        for z, zona in enumerate(zonas):
            quietos[z] = 0 if z in acesas else quietos[z] + 1
            if z in acesas and eventos[z] is None:
                dela = [c for c in caixas if toca(c, zona)]
                eventos[z] = comeca(z, cru, dela, fps)
            if eventos[z] is not None and eventos[z]["video"] is not None:
                eventos[z]["video"].write(cru)
            if eventos[z] is not None and quietos[z] >= valores["paragem"]:
                acaba(z, eventos[z])
                eventos[z], onde[z], sentido[z] = None, None, 1

        # 5. O que se vê: as zonas acesas, as caixas, as animações, as molduras e o estado
        sombreia_zonas(fotograma, zonas, acesas)
        for x, y, w, h in caixas:
            cv.rectangle(fotograma, (x, y), (x + w, y + h), AMARELO, 2, cv.LINE_AA)
        for z, zona in enumerate(zonas):
            if quietos[z] < valores["paragem"]:    # esta zona está a animar
                objeto = sequencias[z][i[z] % len(sequencias[z])]
                reage(fotograma, objeto, zona, tipos[z], onde[z], fracao[z])
                i[z] += sentido[z] if tipos[z] == "sentido" else 1
            else:
                i[z] = 0                           # parada: volta ao princípio
        if cabeca is not None:
            x, y, w, h = cabeca
            cv.rectangle(fotograma, (x, y), (x + w, y + h), MAGENTA, 2, cv.LINE_AA)
            etiqueta(fotograma, "cabeca", x, max(0, y - 24), MAGENTA)
        for x, y, w, h in maos:
            cv.rectangle(fotograma, (x, y), (x + w, y + h), LARANJA, 2, cv.LINE_AA)
            etiqueta(fotograma, "mao", x, max(0, y - 24), LARANJA)
        desenha_zonas(fotograma, zonas, tipos, acesas)
        if ver_mascara and mascara is not None:
            miniatura(fotograma, mascara)
        desenha_estado(fotograma, sum(e is not None for e in eventos), valores["detecao"],
                       1 - aquecer / AQUECIMENTO)
        cv.imshow(JANELA, fotograma)

        # 6. As teclas
        tecla = cv.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("g"):
            CONFIG.write_text(json.dumps(valores, indent=2) + "\n", encoding="utf-8")
            consola.log("gravado em", CONFIG, valores)
        if tecla in (ord("d"), ord("f")):
            if tecla == ord("d"):                  # diferença -> MOG2 -> KNN -> diferença
                n = (DETECOES.index(valores["detecao"]) + 1) % len(DETECOES)
                valores["detecao"] = DETECOES[n]
            subtrator = cria_subtrator(valores["detecao"])   # o fundo aprende-se do zero
            anterior, aquecer = None, AQUECIMENTO
            consola.log(f"[cyan]deteção: {valores['detecao']}[/], a aprender o fundo")
        if tecla == ord("p"):                      # cabeça e mãos, ou não
            ver_pele = not ver_pele
        if tecla == ord("v"):                      # a máscara no canto, ou não
            ver_mascara = not ver_mascara
finally:
    # 7. Ao sair, com q, com Ctrl+C ou num erro: fechar os eventos, a câmara e as janelas
    for z, evento in enumerate(eventos):
        if evento is not None:
            acaba(z, evento)
    camara.release()
    cv.destroyAllWindows()
