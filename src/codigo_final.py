#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "opencv-python>=5.0,<6",
#   "rich",
# ]
# ///
"""Aplicação de visão computacional: a cena reage ao movimento.

    uv run teste_estavel.py [pasta dos assets] [índice da câmara ou URL DroidCam]

Usa KNN para detetar movimento em cada zona, anima PNG com transparência e grava
evidências MP4 e registos JSON. As reações e zonas são definidas em config.json.

Teclas: p parar · c caixas · f reiniciar fundo · v máscara · q sair
"""
import json
import pathlib
import sys
from datetime import datetime

import cv2 as cv
import numpy as np
from rich.console import Console

# --- Configuração --------------------------------------------------------------------------------

raiz = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "../assets")
fonte = sys.argv[2] if len(sys.argv) > 2 else "0"
fonte = int(fonte) if fonte.isdigit() else fonte     # 0, 1, ... ou "http://IP:4747/video"

CONFIG = pathlib.Path("config.json")
EVIDENCIAS = pathlib.Path("../evidencias")
REGISTO = pathlib.Path("registo.json")
TIPOS = ("normal", "movimento", "sentido", "tamanho")

# Parâmetros de deteção fixos: deixa de ser necessário afiná-los com o rato.
# 25 corresponde a um dist2Threshold de 400 no KNN (25 * 16).
NOMES_ZONAS = ["Esquerda", "Meio", "Direita"]
LIMIAR_FIXO = 35
PERCENTAGEM_ZONA_FIXA = 15.0
METODO_DETECAO = "knn"          # para usar MOG2, alterar apenas esta constante para "mog2"

OMISSAO = {"limiar": LIMIAR_FIXO, "percentagem": PERCENTAGEM_ZONA_FIXA,
           "detecao": METODO_DETECAO,
           "zonas": [[0, 0, 213, 480], [213, 0, 213, 480],
                     [426, 0, 214, 480]],
           "tipos": ["movimento", "sentido", "tamanho"]}
LADO = 160                     # lado, em píxeis, a que se reduzem os PNG
AREA_MINIMA = 700              # ignora manchas pequenas, mas conserva movimento das mãos
JUNTAR = 3                     # fecho moderado, aplicado separadamente em cada zona
APRENDIZAGEM = 0.001           # atualização lenta do fundo durante o funcionamento
AQUECIMENTO = 90               # fotogramas iniciais para aprender o fundo sem acender zonas
TEMPO_EXTRA_GRAVACAO = 3.0      # segundos que o vídeo continua depois de deixar de detetar movimento
CONFIRMACAO_FRAMES = 2         # exige deteção em 2 frames seguidos para ativar uma zona
ELIPSE = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
AMARELO, VERMELHO = (0, 255, 255), (0, 0, 255)
BRANCO, PRETO, CINZENTO = (255, 255, 255), (0, 0, 0), (190, 190, 190)
CORES = [(255, 191, 0), (113, 204, 46), (94, 197, 255),    # uma cor por zona, em BGR: azul,
         (180, 105, 255), (0, 215, 255)]                   # verde, laranja, rosa e dourado
JANELA = "esplanada"
ESPELHO = True                 # vira a imagem na horizontal: quem está à frente vê-se num espelho

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
    # Estes três parâmetros não são lidos do config.json: ficam fixos no código.
    # Assim, um config antigo não volta a colocar, por exemplo, a zona nos 42%.
    valores["limiar"] = LIMIAR_FIXO
    valores["percentagem"] = PERCENTAGEM_ZONA_FIXA
    valores["detecao"] = METODO_DETECAO
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


# --- A deteção do movimento ----------------------------------------------------------------------

def cria_subtrator(detecao):
    """Cria o modelo estatístico de fundo escolhido no código: MOG2 ou KNN."""
    if detecao == "mog2":
        return cv.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True)
    if detecao == "knn":
        return cv.createBackgroundSubtractorKNN(
            history=500, dist2Threshold=400.0, detectShadows=True)
    raise ValueError(f"Método de deteção desconhecido: {detecao}")


def mascara_fundo(subtrator, fotograma, aprendizagem, limiar):
    """Subtração de fundo com MOG2 ou KNN (desafio 1 da PL05): a branco, o que não é fundo.

    O limiar fixo controla a sensibilidade do método escolhido.
    """
    # MOG2 compara cada píxel com um modelo estatístico do fundo; KNN usa vizinhos.
    # O limiar fixo ajusta o parâmetro próprio de cada método.
    if isinstance(subtrator, cv.BackgroundSubtractorMOG2):
        subtrator.setVarThreshold(max(1, limiar))
    else:
        subtrator.setDist2Threshold(max(1, limiar) * 16.0)   # 25 x 16 = 400, valor inicial
    bruta = subtrator.apply(fotograma, learningRate=aprendizagem)
    _, so_objeto = cv.threshold(bruta, 200, 255, cv.THRESH_BINARY)   # as sombras (127) não contam
    so_objeto = cv.morphologyEx(so_objeto, cv.MORPH_OPEN, ELIPSE)     # apaga os pontos soltos
    return cv.dilate(so_objeto, None, iterations=2)                   # junta as manchas


def zonas_com_movimento(mascara, zonas, percentagem, caixas):
    """Determina as zonas ativas, a intensidade e o centro do movimento em cada zona.

    O requisito pede que o centro de uma região detetada pertença à zona. Por isso,
    a ativação usa o centro das caixas delimitadoras, e não apenas qualquer pixel
    branco dentro da zona. A percentagem continua a servir de filtro de sensibilidade.
    """
    acesas = set()
    fracao = [0.0] * len(zonas)
    centros = [None] * len(zonas)

    # Primeiro calcula-se a percentagem de píxeis em movimento em cada zona.
    for z, (x, y, w, h) in enumerate(zonas):
        fatia = mascara[y:y + h, x:x + w]
        fracao[z] = cv.countNonZero(fatia) / (w * h)

    # Só se atribui uma região à zona onde está o centro da respetiva caixa.
    for bx, by, bw, bh in caixas:
        centro_caixa = (bx + bw // 2, by + bh // 2)
        for z, (x, y, w, h) in enumerate(zonas):
            cx, cy = centro_caixa
            if x <= cx < x + w and y <= cy < y + h:
                if fracao[z] > percentagem / 100:
                    acesas.add(z)
                    if centros[z] is None:
                        centros[z] = []
                    centros[z].append(centro_caixa)

    # Se houver várias regiões na mesma zona, usa-se o centro médio dessas regiões.
    for z in acesas:
        xs = [p[0] for p in centros[z]]
        ys = [p[1] for p in centros[z]]
        centros[z] = (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

    return acesas, fracao, centros


def caixas_do_movimento(mascara, zonas):
    """Deteta caixas de movimento separadamente em cada zona da imagem.

    Se os contornos forem calculados na imagem inteira, uma região larga (por exemplo,
    uma mão muito perto da câmara) pode atravessar várias colunas. O centro da caixa global
    pode cair apenas na coluna central. Ao recortar cada zona antes de procurar contornos,
    cada coluna consegue identificar a parte do movimento que realmente aparece nela.
    As coordenadas locais dos contornos são depois convertidas para coordenadas da imagem.
    """
    caixas = []
    for x, y, w, h in zonas:
        recorte = mascara[y:y + h, x:x + w]

        # O fecho morfológico une pequenas falhas dentro da zona sem juntar as três colunas.
        juntas = cv.morphologyEx(
            recorte, cv.MORPH_CLOSE, ELIPSE, iterations=JUNTAR)
        contornos, _ = cv.findContours(
            juntas, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contorno in contornos:
            if cv.contourArea(contorno) >= AREA_MINIMA:
                bx, by, bw, bh = cv.boundingRect(contorno)
                caixas.append((x + bx, y + by, bw, bh))

    return caixas


# --- O desenho e as reações ----------------------------------------------------------------------

def compoe(fundo, objeto, x, y):
    """Mistura por alfa do objeto (BGRA) no fundo, com o canto superior esquerdo em (x, y)."""
    h, w = objeto.shape[:2]
    x = max(0, min(x, fundo.shape[1] - w))       # não deixar sair da imagem: encosta
    y = max(0, min(y, fundo.shape[0] - h))
    fatia = fundo[y:y + h, x:x + w]
    if fatia.shape[:2] != (h, w):                # objeto maior do que a própria imagem
        return
    # O canal alfa indica a opacidade de cada píxel (0 = transparente, 255 = opaco).
    # Converte-se para 0..1 e aplica-se a mistura: objeto*alfa + fundo*(1-alfa).
    alfa = objeto[:, :, 3:4].astype(np.float32) / 255.0
    mistura = alfa * objeto[:, :, :3] + (1.0 - alfa) * fatia
    fundo[y:y + h, x:x + w] = mistura.astype(np.uint8)


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
        etiqueta(imagem, NOMES_ZONAS[z], x + 6, y + 6, cor)


def desenha_caixas(imagem, caixas):
    """Cada caixa de movimento só pelos cantos: marca a região sem tapar a imagem."""
    for x, y, w, h in caixas:
        c = max(10, min(w, h) // 5)                # o comprimento de cada canto
        for px, py, dx, dy in ((x, y, 1, 1), (x + w, y, -1, 1),
                               (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
            cv.line(imagem, (px, py), (px + dx * c, py), AMARELO, 2, cv.LINE_AA)
            cv.line(imagem, (px, py), (px, py + dy * c), AMARELO, 2, cv.LINE_AA)



def miniatura(imagem, mascara):
    """A máscara do movimento em pequeno, no canto de baixo à direita (tecla v)."""
    altura, largura = imagem.shape[:2]
    w, h = largura // 4, altura // 4
    x, y = largura - w - 10, altura - h - 40
    imagem[y:y + h, x:x + w] = cv.cvtColor(cv.resize(mascara, (w, h)), cv.COLOR_GRAY2BGR)
    cv.rectangle(imagem, (x - 1, y - 1), (x + w, y + h), BRANCO, 1)


def desenha_estado(imagem, a_gravar, detecao, progresso, parada=False):
    """A barra de baixo, com a deteção (ou o fundo a aprender) e as teclas, e o REC a piscar.

    a_gravar é o número de eventos abertos; progresso vai de 0 a 1 enquanto o fundo se aprende;
    parada diz se a imagem está congelada (a tecla p passa a ser para continuar).
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
    teclas = ("p continuar" if parada else "p parar") + \
        "  c caixas  f reaprender fundo  v mascara  q sair"
    (w, _), _ = cv.getTextSize(teclas, cv.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv.putText(imagem, teclas, (largura - w - 10, altura - 10), cv.FONT_HERSHEY_SIMPLEX, 0.4,
               CINZENTO, 1, cv.LINE_AA)
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
consola.log(
    f"deteção fixa: {METODO_DETECAO.upper()} | limiar: {LIMIAR_FIXO} | "
    f"ativação por zona: {PERCENTAGEM_ZONA_FIXA:g}%")
consola.log(f"{REGISTO}: {len(le_registo())} evento(s) de antes")

cv.namedWindow(JANELA)

camara = cv.VideoCapture(fonte)
if not camara.isOpened():
    sys.exit(f"Não consegui abrir a câmara {fonte}.")
camara.set(cv.CAP_PROP_FRAME_WIDTH, 640)           # as zonas estão em píxeis de 640 x 480
camara.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
fps = camara.get(cv.CAP_PROP_FPS) or 30            # há câmaras que dizem 0
paragem_frames = max(1, round(fps * TEMPO_EXTRA_GRAVACAO))
consola.log(f"gravação prolongada: {TEMPO_EXTRA_GRAVACAO:g} s após o último movimento")

i = [0] * len(zonas)                               # o fotograma da animação de cada zona
quietos = [paragem_frames] * len(zonas)        # fotogramas sem movimento, por zona
confirmacoes = [0] * len(zonas)                    # deteções consecutivas para evitar falsos alertas
eventos = [None] * len(zonas)                      # o evento aberto em cada zona
onde = [None] * len(zonas)                         # o centro do movimento em cada zona
sentido = [1] * len(zonas)                         # +1 para a frente, -1 para trás
subtrator = cria_subtrator(METODO_DETECAO)        # modelo estatístico fixo (KNN por omissão)
aquecer = AQUECIMENTO
ver_mascara = False
ver_caixas = True                                  # as caixas do movimento (tecla c)
congelado = False                                  # a imagem está parada (tecla p)
ultimo = None                                      # a última imagem, para mostrar parada

# --- O ciclo principal ---------------------------------------------------------------------------

try:
    while True:
        if not congelado:
            # 1. Leitura da imagem da câmara e preparação do fotograma.
            ok, fotograma = camara.read()
            if not ok:
                break
            if ESPELHO:
                fotograma = cv.flip(fotograma, 1)      # 1: vira na horizontal
            altura, largura = fotograma.shape[:2]
            confere(zonas, largura, altura)
            cru = carimba(fotograma)                   # a evidência: sem desenhos, com a hora

            # 2. A máscara do movimento
            taxa = -1 if aquecer > 0 else APRENDIZAGEM     # no arranque aprende o fundo depressa
            mascara = mascara_fundo(subtrator, fotograma, taxa, valores["limiar"])
            if aquecer > 0:                            # a aprender o fundo: nenhuma zona acende
                aquecer -= 1
                mascara = None

            # 3. As zonas acesas, as caixas e o sentido do movimento em cada zona
            acesas, caixas = set(), []
            fracao, centros = [0.0] * len(zonas), [None] * len(zonas)
            if mascara is not None:
                caixas = caixas_do_movimento(mascara, zonas)
                candidatas, fracao, centros = zonas_com_movimento(
                    mascara, zonas, valores["percentagem"], caixas)
            else:
                candidatas = set()

            # Duas confirmações consecutivas reduzem ativações causadas por ruído isolado.
            acesas = set()
            for z in range(len(zonas)):
                confirmacoes[z] = confirmacoes[z] + 1 if z in candidatas else 0
                if confirmacoes[z] >= CONFIRMACAO_FRAMES:
                    acesas.add(z)

            for z in acesas:
                if onde[z] is not None and abs(centros[z][0] - onde[z][0]) > 3:
                    sentido[z] = 1 if centros[z][0] > onde[z][0] else -1
                onde[z] = centros[z]

            # 4. Os eventos: começam quando a zona acende e acabam quando ela sossega
            for z, zona in enumerate(zonas):
                # O contador reinicia com movimento; sem movimento, avança até "paragem".
                # Assim, a animação/evento não termina por uma pequena falha de deteção.
                # Mantém o vídeo aberto durante TEMPO_EXTRA_GRAVACAO após o movimento.
                quietos[z] = 0 if z in acesas else quietos[z] + 1
                if z in acesas and eventos[z] is None:
                    dela = [c for c in caixas if toca(c, zona)]
                    eventos[z] = comeca(z, cru, dela, fps)
                if eventos[z] is not None:
                    # O JSON conserva as caixas observadas ao longo de todo o evento.
                    for caixa in caixas:
                        if toca(caixa, zona) and caixa not in eventos[z]["caixas"]:
                            eventos[z]["caixas"].append(caixa)
                    if eventos[z]["video"] is not None:
                        eventos[z]["video"].write(cru)
                if eventos[z] is not None and quietos[z] >= paragem_frames:
                    acaba(z, eventos[z])
                    eventos[z], onde[z], sentido[z] = None, None, 1

            # 5. O que se vê: as zonas acesas, as caixas, as animações, as molduras e o estado
            sombreia_zonas(fotograma, zonas, acesas)
            if ver_caixas:
                desenha_caixas(fotograma, caixas)
            for z, zona in enumerate(zonas):
                if quietos[z] < paragem_frames:    # esta zona está a animar
                    objeto = sequencias[z][i[z] % len(sequencias[z])]
                    reage(fotograma, objeto, zona, tipos[z], onde[z], fracao[z])
                    i[z] += sentido[z] if tipos[z] == "sentido" else 1
                else:
                    i[z] = 0                           # parada: volta ao princípio
            desenha_zonas(fotograma, zonas, tipos, acesas)
            if ver_mascara and mascara is not None:
                miniatura(fotograma, mascara)
            ultimo = fotograma.copy()                  # é esta que fica, se a imagem parar
            desenha_estado(fotograma, sum(e is not None for e in eventos), valores["detecao"],
                           1 - aquecer / AQUECIMENTO)
            ecra = fotograma
        else:
            # Imagem parada: a câmara continua a ser lida e deitada fora, para não atrasar.
            # Mantém-se a última imagem, sem apresentar um aviso sobre a pausa.
            if not camara.grab():
                break
            ecra = ultimo.copy()
            desenha_estado(ecra, 0, valores["detecao"], 1, parada=True)
        cv.imshow(JANELA, ecra)

        # 6. As teclas
        tecla = cv.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("p"):                      # parar a imagem, ou pô-la outra vez a correr
            congelado = not congelado
            if congelado:                          # os eventos abertos acabam aqui
                for z, evento in enumerate(eventos):
                    if evento is not None:
                        acaba(z, evento)
                        eventos[z], onde[z], sentido[z] = None, None, 1
                        confirmacoes[z] = 0
                consola.log("[magenta]imagem parada[/]")
            else:
                consola.log("[magenta]imagem outra vez a correr[/]")
        if tecla == ord("f"):
            # Fecha eventos em curso antes de reiniciar a aprendizagem do fundo.
            for z, evento in enumerate(eventos):
                if evento is not None:
                    acaba(z, evento)
                    eventos[z] = None
            quietos = [paragem_frames] * len(zonas)
            onde = [None] * len(zonas)
            sentido = [1] * len(zonas)
            confirmacoes = [0] * len(zonas)
            subtrator = cria_subtrator(METODO_DETECAO)
            aquecer = AQUECIMENTO
            consola.log("[cyan]fundo reiniciado[/], a reaprender o cenário")
        if tecla == ord("v"):                      # a máscara no canto, ou não
            ver_mascara = not ver_mascara
        if tecla == ord("c"):                      # as caixas do movimento, ou não
            ver_caixas = not ver_caixas
finally:
    # 7. Ao sair, com q, com Ctrl+C ou num erro: fechar os eventos, a câmara e as janelas
    for z, evento in enumerate(eventos):
        if evento is not None:
            acaba(z, evento)
    camara.release()
    cv.destroyAllWindows()
