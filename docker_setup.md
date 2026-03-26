# Docker Desktop Setup

This guide will walk you through installing and configuring Docker Desktop for your operating system.

## 🍎 For Mac

1. **Download**: Go to [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/) and click **Download for Mac** (choose between Apple Silicon or Intel depending on your hardware).
2. **Install**: Open the `.dmg` file and drag the Docker icon to your **Applications** folder.
3. **Launch**: Open Docker Desktop from your Applications folder.
4. **Setup**: Accept the terms and complete the initial setup wizard.
5. **Verify**: Open your terminal and run:
   ```bash
   docker --version
   ```

## 🪟 For Windows

1. **Download**: Go to [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/) and click **Download for Windows**.
2. **Install**: Run the downloaded `.exe` installer.
3. **WSL 2**: Ensure "Enable WSL 2 features" is checked when prompted. This is critical for performance and compatibility.
4. **Restart**: Restart your PC after the installation completes.
5. **Launch**: Open Docker Desktop from the Start menu.
6. **Verify**: Open PowerShell or Command Prompt and run:
   ```bash
   docker --version
   ```

## 🔗 References
- [Official Docker Installation Guide](https://docs.docker.com/desktop/)
- [Docker Desktop Windows Install Guide](https://docs.docker.com/desktop/setup/install/windows-install/)
- [Docker Desktop Mac Install Guide](https://docs.docker.com/desktop/setup/install/mac-install/)
