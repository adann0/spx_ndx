# Setup

## Clone et environnement

```bash
git clone https://github.com/adann0/spx_ndx
cd spx_ndx

python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
make test
make fetch
make plot

make dataset
make consensus
make stresstest
make plotmodel
```
