### Удаление предыдущей версии

```bash

# драйвера видеокарты удаляются тоже
sudo apt-get --purge remove "*cublas*" "*cuda*" "*nvidia*"
sudo apt autoremove

```

### Установка драйверов видеокарты

```bash
# определение подходящей версии
ubuntu-drivers devices
# установка
sudo apt install <подходящая версия>
```

```bash

sudo apt-key del 7fa2af80  # Удалите старый ключ (если нужно)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d '.')/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

```

## Установка cuda

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_network