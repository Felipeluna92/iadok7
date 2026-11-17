# FaceSwap Studio

Troca de rosto em vídeo com foco em **qualidade**, **movimento suave** e **enquadramento estável**.
Roda localmente na sua GPU NVIDIA (testado para GTX 1660 Super, 6 GB) com interface web.

![Interface](docs/interface.png)

## Como funciona

| Etapa | Modelo / técnica | Para quê |
|---|---|---|
| Análise da referência | SCRFD + ArcFace + idade/gênero + segmentação | Nota de qualidade de cada foto (resolução, nitidez, ângulo), descarte de fotos de outra pessoa, tom de pele |
| Identidade | Média ponderada pela qualidade | As melhores fotos pesam mais na identidade final |
| Compatibilidade | Referência × vídeo | Avisa sobre gênero, idade, resolução e tom de pele incompatíveis antes de processar |
| Rastreamento | IoU + identidade | Segue a pessoa entre frames, inclusive depois de cortes |
| **Suavização** | Mediana + Gaussiana **bidirecional** nos pontos do rosto | Elimina tremido sem atrasar o movimento; preenche frames sem detecção |
| **Troca** | **HyperSwap 1C** (256 px nativo) + pixel boost para 512 px | Traços da referência (olhos, sobrancelhas, nariz, lábios) com detalhe real |
| Força da identidade | Afasta a identidade do rosto original | Remove traços que "vazam" do rosto original do vídeo |
| Máscara | Borda suave + segmentação BiSeNet + média entre frames | Bordas invisíveis; mãos e objetos na frente do rosto ficam por cima |
| **Realismo** | Nitidez e granulação iguais às da câmera; estabilização temporal adaptativa | O rosto ganha a mesma textura, foco e ruído do vídeo, sem "cara de IA" nem "fervura" |
| Tom de pele (opcional) | Transferência de cor LAB preservando a luz | Aproxima a cor da pele da referência sem mudar a iluminação da cena |
| Saída | FFmpeg H.264 (CRF 14, preset slow) + áudio original | Arquivo final com qualidade alta |

### Testes feitos

Comparação de modelos (semelhança ArcFace com a referência / com o rosto original, 6 rostos):
HyperSwap 1C ficou visualmente mais fiel aos traços da referência e mais natural do que o InSwapper.
A força de identidade 0,2 reduziu a semelhança com o rosto original de ~0,10 para ~0,00 sem perder
semelhança com a referência. A granulação automática levou o ruído do rosto de 0,78 para 1,72,
contra 1,77 no resto da cena.

### O que vem da referência e o que vem do vídeo

A troca transfere **os traços do rosto**: olhos, sobrancelhas, nariz, boca, lábios e textura da pele.
O que fica do vídeo: **formato da cabeça e do queixo, cabelo, orelhas, pescoço, expressão,
iluminação e movimento**. Para o resultado mais convincente, escolha um vídeo de alguém com
gênero, idade aproximada, formato de rosto e cabelo parecidos com a pessoa da referência — é o que a
análise de compatibilidade verifica.

## Instalação (Windows)

1. Atualize o driver da NVIDIA (versão 580 ou mais recente, exigida pelo CUDA 13).
2. Instale o **Python 3.11 ou 3.12** em python.org, marcando "Add python.exe to PATH".
3. Baixe este repositório e dê dois cliques em **`instalar.bat`**.
   No final deve aparecer `CUDAExecutionProvider` na lista de provedores.
4. Dê dois cliques em **`iniciar.bat`**. O navegador abre em http://127.0.0.1:7860.

Na primeira execução os modelos necessários (cerca de 0,8 GB na configuração padrão) são baixados para a pasta `models/`.
Os vídeos prontos ficam em `outputs/`.

**Se a GPU não aparecer:** como alternativa, troque o CUDA pelo DirectML, que funciona com qualquer
GPU no Windows sem instalar CUDA (é um pouco mais lento):

```bat
venv\Scripts\activate
pip uninstall -y onnxruntime-gpu
pip install onnxruntime-directml
```

## Uso

1. **Fotos de referência**: 3 a 10 fotos nítidas da mesma pessoa, com o rosto grande na foto (de
   preferência 1000 px ou mais), ângulos variados (frente e levemente de lado) e expressões variadas.
2. **Vídeo alvo**: o vídeo da pessoa cujo rosto será trocado.
3. **Analisar**: confira a nota de cada foto e os avisos de compatibilidade. Remova fotos com nota baixa.
4. **Prévia**: escolha um momento difícil do vídeo (perfil, movimento rápido, mão no rosto) e compare
   antes/depois com a barra. A prévia mostra a **fidelidade** (semelhança com a referência).
5. **Processar**: no fim aparece a fidelidade média e a mínima do vídeo inteiro.

### Ajuste fino

- **Fidelidade baixa (< 0,6)**: suba a "Força da identidade" para 0,3 e use fotos melhores.
- **Rosto parece "colado" ou limpo demais**: mantenha "Igualar nitidez" ligado e suba a granulação para 1,2.
- **Pele de cor diferente**: use o valor sugerido pela análise em "Tom de pele da referência".
- **Tremido**: suba a "Suavização do movimento" para 2,5–3. Se o rosto escorregar em movimentos
  rápidos, baixe para 0,8–1.
- **Testa ou queixo original aparecendo**: ajuste os recuos da máscara.
- **Vídeo de baixa qualidade/borrado**: experimente GFPGAN ou CodeFormer com intensidade 0,3–0,5.
  Em vídeos bons, deixe "Nenhum": o melhorador tende a deixar a pele com aspecto plástico.

## Linha de comando

```bat
venv\Scripts\activate
python cli.py -s foto1.jpg foto2.jpg foto3.jpg -t video.mp4 --analyse-only
python cli.py -s foto1.jpg foto2.jpg foto3.jpg -t video.mp4 -o saida.mp4
python cli.py -s foto1.jpg -t video.mp4 -o saida.mp4 --strength 0.3 --skin 0.5 --grain 1.2
```

## Uso responsável

Use apenas com o consentimento das pessoas envolvidas. A interface exige essa confirmação e, por
padrão, marca o vídeo como "Gerado por IA". Não use para enganar, fraudar, difamar ou criar conteúdo
íntimo de ninguém. Em muitos países isso é crime.

## Créditos dos modelos

InsightFace (SCRFD, ArcFace, idade/gênero, InSwapper — uso não comercial), HyperSwap (FaceFusion), GFPGAN, GPEN, CodeFormer, BiSeNet.
Versões ONNX distribuídas pelo projeto FaceFusion. Verifique as licenças de cada modelo antes de
qualquer uso comercial.
