# Install

First add the required data files in the right directories `./data/raw/CustomerCenter 1 1.e57` and `./data/json/objects2.example.json`

## Docker

Currently, running in Docker does not support displaying any plots.

```
docker-compose up
```

## Manual

```
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### xerces

You might have to install xerces
([see building notes of pye57](https://pypi.org/project/pye57/))
if you are on Windows or Linux.
It should work out-of-the-box for macOS.

* Conda users on **Windows**: ```conda install xerces-c```

* **Linux** users can just use their package repository: 

  * Debian, Ubuntu based systems: ```apt install libxerces-c-dev```

  * Fedora, Red Hat, CentOS based systems: ```dnf install xerces-c-devel```

# Credits

Created as part of the Hackdays Baden-Württemberg 2020.

* [akonnova](https://www.behance.net/akonnova)
* [Dinozaver959](https://github.com/Dinozaver959)
* [hajerchebil](https://github.com/hajerchebil)
* [quassy](https://github.com/quassy)
