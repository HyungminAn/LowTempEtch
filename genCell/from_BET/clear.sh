find . -mindepth 2 -maxdepth 2 -type f ! -name 'input.yaml' ! -name 'mol.xyz' -exec rm -f {} \;
