{
  description = "Anna's Archive download microservice";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          name = "annas-dl";

          packages = with pkgs; [
            # uv manages Python itself
            uv

            # For native extensions (pydantic-core is Rust)
            rustc
            cargo
            pkg-config
            openssl
            openssl.dev

          ];

          shellHook = ''
            # Let uv handle Python 3.14t
            if [ ! -d .venv ]; then
              echo "Creating venv with Python 3.13 free-threaded..."
              uv venv --python 3.13t
            fi

            source .venv/bin/activate

            echo "🐍 $(python --version) | GIL disabled: $(python -c 'import sys; print(not sys._is_gil_enabled())')"

            uv sync --quiet 2>/dev/null || true
          '';

          # For pydantic-core Rust compilation
          CARGO_NET_GIT_FETCH_WITH_CLI = "true";
        };
      }
    );
}
