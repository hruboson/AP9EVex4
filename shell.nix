{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python314.withPackages (ps: with ps; [
    matplotlib
    pyside6  # For Qt bindings
  ]);
in

pkgs.mkShell {
  name = "python-venv-shell";

  packages = [
    python
    pkgs.pkg-config
    pkgs.qt6.qtbase  # For matplotlib Qt backend
    pkgs.libx11
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.libz
    pkgs.libx11
  ];

shellHook = ''
  echo "# Generated from nix-shell on $(date)" > requirements.txt
  echo "# Run: pip install -r requirements.txt" >> requirements.txt
  echo "" >> requirements.txt

  ${python}/bin/python - <<EOF >> requirements.txt
import importlib.metadata

# Exclude these since pip/setuptools/wheel are universal
excluded = {"pip", "setuptools", "wheel"}

packages = []
for dist in importlib.metadata.distributions():
    name = dist.metadata["Name"]
    version = dist.version
    if name and name not in excluded:
        packages.append(f"{name}=={version}")

for pkg in sorted(packages):
    print(pkg)
EOF

  echo "📦 Generated requirements.txt with $(grep -c '==' requirements.txt || echo 0) packages"
  echo "🐍 Python environment ready: $(python --version)"
'';
}
