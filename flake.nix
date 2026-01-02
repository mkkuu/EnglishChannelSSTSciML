{
  description = "m1Project - SciML SST Copernicus";

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
          plotly
          dash-bootstrap-components
          curl-cffi
          requests
          certifi
          lxml
          ipykernel
          copernicusmarine
          xarray
          netcdf4
          scipy
          jupyterlab
        ]);
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [ pythonEnv ];
          shellHook = ''
            export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
            export REQUESTS_CA_BUNDLE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
          '';
        };
      });
}
