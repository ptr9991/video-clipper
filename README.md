# Video Clipper

Transforme vídeos longos em clipes curtos de 30–50 segundos com IA.

**Todo o processamento de vídeo é local.** Só o áudio é enviado à API da Groq para transcrição e análise.

---

## Para usuário normal (Windows)

Você **não precisa** instalar Python, FFmpeg, Git nem abrir o terminal.

### 1. Baixe o instalador

Vá em **[Releases](https://github.com/ptr9991/video-clipper/releases)** e baixe o arquivo:

**`VideoClipperSetup.exe`**

(Se ainda não houver release, o instalador também aparece como *Artifact* na aba Actions após um build bem-sucedido.)

### 2. Instale

1. Clique duas vezes em `VideoClipperSetup.exe`
2. Siga as telas (pode deixar as opções padrão)
3. Marque “Criar atalho na Área de Trabalho” se quiser
4. Clique em Instalar

### 3. Abra o Video Clipper

Clique no atalho **Video Clipper**.

Na primeira vez aparecerá uma janela pedindo a **Groq API Key**:

1. Crie uma chave grátis em https://console.groq.com/keys
2. Cole a chave na janela
3. (Opcional) clique em **Testar conexão**
4. Clique em **Continuar**

O navegador abrirá automaticamente com o Video Clipper.

### 4. Use

1. Faça upload de um vídeo (MP4, MOV, MKV…)
2. Clique em **Encontrar melhor clipe com IA**
3. Ajuste os tempos se quiser
4. Clique em **Gerar clipe** e baixe o MP4

Pronto. Nenhuma outra configuração é necessária.

---

## Para desenvolvedores

### Arquitetura

```
video-clipper/
├── app.py                      # Interface Streamlit
├── requirements.txt
├── src/                        # Lógica (transcrição, análise, FFmpeg)
├── scripts/
│   ├── launcher.py             # Launcher Windows (GUI de API key + Streamlit)
│   ├── prepare_portable.ps1    # Monta a pasta portable (Python + FFmpeg + app)
│   └── VideoClipper.bat        # Atalho de desenvolvimento
├── installer/
│   └── VideoClipper.iss        # Script Inno Setup → VideoClipperSetup.exe
├── tests/
└── .github/workflows/
    └── build-windows-installer.yml
```

### Como o instalador funciona

1. **GitHub Actions** (Windows) roda os testes.
2. O script `prepare_portable.ps1`:
   - Baixa o **Python embeddable oficial** (python.org)
   - Instala `pip` + dependências do `requirements.txt`
   - Baixa **FFmpeg essentials** de https://www.gyan.dev/ffmpeg/builds/ (fonte linkada pelo site oficial do FFmpeg)
   - Copia o código do app + cria `VideoClipper.bat`
3. **Inno Setup** empacota tudo em `VideoClipperSetup.exe`.
4. O instalador grava em `%LOCALAPPDATA%\VideoClipper` (não precisa de administrador).
5. O atalho chama o launcher, que:
   - Pede a API key (salva só neste PC, em `%LOCALAPPDATA%\VideoClipper\settings.json`)
   - Aponta para o FFmpeg embutido
   - Sobe o Streamlit sem janela de terminal
   - Abre o navegador automaticamente

### Fontes oficiais dos downloads (build)

| Componente | Fonte |
|------------|--------|
| Python embeddable | https://www.python.org/ftp/python/ |
| get-pip.py | https://bootstrap.pypa.io/get-pip.py |
| FFmpeg essentials | https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip |
| Inno Setup | https://jrsoftware.org/isinfo.php (via Chocolatey no CI) |

Nenhum executável de site desconhecido é baixado.

### Rodar a partir do código-fonte (desenvolvimento)

```bash
git clone https://github.com/ptr9991/video-clipper.git
cd video-clipper
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
# FFmpeg precisa estar no PATH ou defina FFMPEG_PATH
streamlit run app.py
# ou: python scripts/launcher.py
```

### Testes

```bash
pip install pytest
pytest tests/ -v
```

### Gerar o instalador localmente (Windows)

1. Instale [Inno Setup 6](https://jrsoftware.org/isinfo.php)
2. Abra PowerShell na raiz do repositório:
   ```powershell
   .\scripts\prepare_portable.ps1
   & "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\VideoClipper.iss
   ```
3. O instalador sairá em `dist\VideoClipperSetup.exe`

### Publicar uma nova versão

```bash
git tag v1.0.1
git push origin v1.0.1
```

O workflow cria automaticamente um **GitHub Release** com o `VideoClipperSetup.exe`.

### Modelo de corte

- **Padrão (rápido)**: `-c copy` (sem re-encode)
- **Opcional (preciso)**: re-encode com libx264

### Limitações

- O modo `-c copy` alinha o início a keyframes (pode variar alguns frames).
- A qualidade da seleção depende da transcrição e do conteúdo falado.
- Rate limits da conta gratuita da Groq se aplicam.

---

## Licença

Uso pessoal e educacional. Use por sua conta e risco.
