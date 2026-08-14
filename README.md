# AI Video Clipper Local

Aplicação **100% local** que transforma vídeos longos em clipes curtos de 30–50 segundos usando:

- **Streamlit** – interface web local
- **FFmpeg** – extração de áudio e corte de vídeo (processamento local)
- **Groq API** – apenas para transcrição (Whisper) e análise textual do melhor trecho

O vídeo original **nunca** sai da sua máquina. Somente o áudio otimizado é enviado à API da Groq.

---

## Requisitos

| Componente       | Versão / Observação                          |
|------------------|----------------------------------------------|
| Python           | 3.10 ou superior                             |
| FFmpeg           | Instalado no sistema e disponível no PATH    |
| Conta Groq       | Chave de API gratuita em https://console.groq.com |
| Sistema operacional | Windows 10/11, macOS ou Linux             |

---

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/ptr9991/video-clipper.git
cd video-clipper
```

### 2. Crie e ative um ambiente virtual

**Windows (CMD)**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instale as dependências Python

```bash
pip install -r requirements.txt
```

> **Importante:** FFmpeg **não** é um pacote Python. Ele deve ser instalado no sistema operacional.

### 4. Configure a chave da Groq

```bash
cp .env.example .env
```

Edite `.env` e coloque sua chave:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```

Alternativas (sem arquivo `.env`):

**Windows PowerShell**
```powershell
$env:GROQ_API_KEY="sua_chave"
```

**Windows CMD**
```cmd
set GROQ_API_KEY=sua_chave
```

**macOS / Linux**
```bash
export GROQ_API_KEY="sua_chave"
```

### 5. Instale o FFmpeg

#### Windows

1. Baixe o build em https://www.gyan.dev/ffmpeg/builds/ (versão `ffmpeg-release-essentials.zip`)
2. Extraia para uma pasta, por exemplo `C:\ffmpeg`
3. Adicione `C:\ffmpeg\bin` ao PATH do sistema
4. Abra um **novo** terminal e teste:

```cmd
ffmpeg -version
```

Ou use o gerenciador de pacotes:

```powershell
winget install ffmpeg
```

#### macOS

Com Homebrew (recomendado):

```bash
brew install ffmpeg
```

#### Linux (Debian/Ubuntu)

```bash
sudo apt update
sudo apt install ffmpeg
```

#### Verificação

```bash
ffmpeg -version
ffprobe -version
```

Se o FFmpeg estiver em um local customizado, defina:

```
FFMPEG_PATH=/caminho/completo/para/ffmpeg
```

---

## Executar a aplicação

```bash
streamlit run app.py
```

O navegador abrirá automaticamente em `http://localhost:8501`.

---

## Fluxo de uso

1. Faça upload de um vídeo (MP4, MOV, MKV ou WEBM).
2. Visualize o preview e os metadados (duração, resolução, tamanho).
3. Clique em **“Encontrar melhor clipe com IA”**.
4. Aguarde:
   - Extração local do áudio (FFmpeg)
   - Transcrição via Groq Whisper (`whisper-large-v3-turbo`)
   - Análise do texto por um modelo de linguagem Groq (`llama-3.3-70b-versatile`)
5. Veja o trecho sugerido (início, fim, duração, score e motivo).
6. Ajuste os timestamps com sliders ou campos numéricos (máx. 50 s).
7. Escolha o modo de corte:
   - **Rápido** (`-c copy`) – padrão, quase instantâneo, alinhado a keyframes
   - **Preciso** (re-encode) – frame-accurate, mais lento
8. Clique em **“Gerar clipe”** e baixe o MP4.

---

## Arquitetura

```
video-clipper/
├── app.py                  # Interface Streamlit e orquestração
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py           # Variáveis de ambiente, caminhos, modelos
│   ├── transcription.py    # Extração de áudio + chamada Groq Whisper
│   ├── clip_analyzer.py    # Prompt + análise LLM + validação JSON
│   ├── video_processor.py  # Metadados (ffprobe) + corte FFmpeg
│   └── utils.py            # Helpers (timestamps, JSON, arquivos temp)
├── temp/                   # Arquivos temporários (áudio, uploads)
├── output/                 # Clipes gerados
└── tests/
    ├── test_utils.py
    ├── test_clip_analyzer.py
    └── test_video_processor.py
```

---

## Detalhes técnicos importantes

### Corte rápido vs preciso

| Modo     | Comando principal                         | Velocidade | Precisão          |
|----------|-------------------------------------------|------------|-------------------|
| **fast** | `-ss START -i INPUT -t DUR -c copy`       | Instantâneo| Keyframe-aligned  |
| **precise** | `-i INPUT -ss START -t DUR -c:v libx264` | Mais lento | Frame-accurate    |

O modo **fast** é o padrão porque:

- Não re-renderiza o vídeo
- Consome quase zero CPU/GPU
- É suficiente para a grande maioria dos casos de uso de clipes curtos

A pequena imprecisão no início (geralmente < 1–2 s) é o preço do stream-copy. O modo preciso existe como opção avançada.

### Modelos utilizados (2026)

- Transcrição: `whisper-large-v3-turbo`
- Análise: `llama-3.3-70b-versatile`

---

## Testes

```bash
pip install pytest
pytest tests/ -v
```

Os testes não fazem chamadas reais à API da Groq nem executam FFmpeg; eles validam parsing de JSON, construção de comandos e lógica de timestamps.

---

## Solução de problemas

| Problema                         | Solução                                              |
|----------------------------------|------------------------------------------------------|
| `FFmpeg not found`               | Instale e adicione ao PATH ou use `FFMPEG_PATH`      |
| `GROQ_API_KEY is not set`        | Crie `.env` ou exporte a variável                    |
| Rate limit / 429                 | Aguarde alguns minutos (tier gratuito tem limites)   |
| Áudio muito grande               | Use vídeos mais curtos ou comprima antes             |
| Corte começa um pouco antes/depois | Use o modo **Preciso**                             |
| Erro de JSON da IA               | Clique novamente em “Encontrar melhor clipe”         |

---

## Limitações conhecidas

- O modo `-c copy` não é frame-perfect (depende de keyframes).
- Vídeos muito longos geram transcrições grandes; o texto é truncado se necessário.
- A qualidade da seleção depende da qualidade da transcrição e do conteúdo falado.
- Rate limits da conta gratuita da Groq se aplicam.

---

## Licença

Código liberado para uso pessoal e educacional. Use por sua conta e risco.
