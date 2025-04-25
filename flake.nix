{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-24.11";
    rust-overlay = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:oxalica/rust-overlay";};
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, ...}@inputs: inputs.flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import inputs.nixpkgs {
        inherit system;
        overlays = [
          inputs.rust-overlay.overlays.default
        ];
        config = {
          allowUnfree = true;
        };
      };
      in
    {
      devShell = pkgs.mkShell rec {
        nativeBuildInputs = with pkgs;[
          cudatoolkit
          cudaPackages.cuda_cudart
          pkg-config
          rust-bin.stable."1.86.0".default
        ];
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath nativeBuildInputs;
      };
  });
}
