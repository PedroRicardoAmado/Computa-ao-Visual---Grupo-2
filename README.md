# Computa-ao-Visual---Grupo-2

## 1 -Introdução

Este trabalho tem como objetivo a aprender e aprofundar o conhecimento em componentes da computação visual.
O Projeto consiste na criação dum ambiente, no Blender, duma esplanada para quatro pessoas e renderizar uma animação de tal esplanada.
Consiste também na criação dum programa em python utilizando a biblioteca OpenCV para a deteção de movimento através da câmara do computador e "desenhar" a animação renderizada no ecrã.

## 2 - Blender

Projeto no Blender e focado em modelação 3D no Blender, explorando diferentes técnicas de criação de objetos, aplicação de materiais, animação por keyframes e renderização.

| Zonas | Objetos  | Quem Modelou |
|---|---|---|
  | Zona 1 | Cadeira, Chão, Guarda-Sol | Alex Stetsenko |
| Zona 2 | Mesa, Garfo, Faca | João Castanheira |
| Zona 3 | Copo, Prato, Vaso | Pedro Amado |

## 3 OpenCV

O Open CV é uma biblioteca para o processamento de imagens e vídeos, cujo neste projeto foi utilizada para captar vídeo duma câmara e detetar movimentos para "desenhar" os PNGs na imagem enquanto grava.

### 3.1 Como Funciona
A imagem da câmara é dividida em três zonas (Esquerda, Meio e Direita). Quando é detetado movimento numa zona, a sequência de PNG correspondente é animada por cima do vídeo e são gravadas evidências do que aconteceu.

  - Deteção de movimento por subtração de fundo (KNN por omissão, MOG2 opcional), com fase de aquecimento para aprender o fundo, filtro de manchas pequenas e confirmação em frames consecutivos para evitar falsos alertas.
  - Zonas configuráveis em config.json, cada uma com o seu tipo de reação:
      - movimento: o objeto segue o local onde a pessoa se mexe;
      - sentido: a animação corre para a frente ou para trás conforme a direção do movimento;
      - tamanho: o objeto cresce com a quantidade de movimento na zona.
  - Evidências automáticas: por cada evento é gravado um vídeo MP4 (com data e hora), uma fotografia do início com as caixas de movimento e uma entrada em registo.json com zona, início, duração, ficheiro, caixas e objeto.
  - Interface visual: zonas coloridas, caixas de movimento, indicador REC, barra de estado e miniatura da máscara.

