{
  description = "m1Project - SciML SST";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        pythonEnv = pkgs.python312.withPackages (ps: with ps; [
          numpy
          pandas
          matplotlib
          yfinance
          dash
          seaborn
          scikit-learn
          plotly
          dash-bootstrap-components
          curl-cffi
          requests
          certifi
          lxml
          ipykernel
          xarray
          netcdf4
          scipy
          jupyterlab
          dask
          statsmodels
        ]);
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.julia-bin
          ];

          shellHook = ''
            export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
            export REQUESTS_CA_BUNDLE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt

            if ! jupyter kernelspec list | grep -q "m1Project"; then
              python -m ipykernel install --user \
                --name m1Project \
                --display-name "Python m1Project"
            fi
          '';
        };
      });
}
