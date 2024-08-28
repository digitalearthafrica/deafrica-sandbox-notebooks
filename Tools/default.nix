{ pkgs ? import <nixpkgs> {} }:

let
  fhs = pkgs.buildFHSUserEnv {
    name = "fhs";

    targetPkgs = pkgs: [
      pkgs.micromamba
      pkgs.gcc_multi
      pkgs.binutils
    ];

    profile = ''
      set -ex

      # Set up micromamba and initialize the shell
      export MAMBA_EXE=$(which micromamba)
      export MAMBA_ROOT_PREFIX=~/micromamba
      eval "$($MAMBA_EXE shell hook --shell=posix --prefix=$MAMBA_ROOT_PREFIX)"

      # Configure an exclusive conda set up
      micromamba config append channels conda-forge
      micromamba config set channel_priority strict

      # Create the required environment:
      if ! test -d $MAMBA_ROOT_PREFIX/envs/deafrica-tools-env; then
          micromamba create --yes -f environment.yml
      fi

      # Activate the environment.
      micromamba activate deafrica-tools-env

      set +e
    '';
  };
in fhs.env