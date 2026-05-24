{
  description = "A basic flake with a shell";
    #inputs.nixpkgs.url = "github:NixOS/nixpkgs/2d02dc9fc19492c35e94c7419567e6991eefc16e";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};
	
	python = pkgs.python3.withPackages (ps: with ps; [
            numpy
	    onnx
	    onnxruntime
          ]);
    in {
      devShells.default = pkgs.mkShell {

        packages = with pkgs; [
	  julia_111-bin
	  python
        ];
	shellHook = ''
	    export PYTHON=${python}/bin/python
	    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
	 '';
      };
    });
}
