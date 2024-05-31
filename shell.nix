{ pkgs ? import <nixpkgs> {}}:
let
  fhs = pkgs.buildFHSUserEnv {
    name = "my-fhs-environment";

    targetPkgs = _: [
      pkgs.micromamba
      pkgs.gcc_multi
      pkgs.binutils
    ];
  
    profile = ''
      set -e
      eval "$(micromamba shell hook --shell=posix)"
      export MAMBA_ROOT_PREFIX=~/micromamba
      if ! test -d $MAMBA_ROOT_PREFIX/envs/deafrica-tools-env; then
          micromamba create --yes -f Tools/conda/environment.yaml
      fi
      micromamba activate deafrica-tools-env
      set +e
    '';
  };
in fhs.env