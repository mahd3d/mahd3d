# Install

```
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Put your .e57 file in ./data/raw/
```

## xerces

You might have to install xerces
([see building notes of pye57](https://pypi.org/project/pye57/))
if you are on Windows or Linux.
It should work out-of-the-box for macOS.

For Conda users on **Windows**:

```conda install xerces-c```

**Linux** users can just use their package registry. 

For Debian and Ubuntu based systems:

```apt install libxerces-c-dev```

For Fedora, Red Hat and CentOS based systems:

```dnf install xerces-c```

# Credits

* [akonnova](https://www.behance.net/akonnova)
* [Dinozaver959](https://github.com/Dinozaver959)
* [hajerchebil](https://github.com/hajerchebil)
* [quassy](https://github.com/quassy)
