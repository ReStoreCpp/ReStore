# ReStore	REplicated STORagE

## Further suggestions for naming
REFUSE	REad-only FaUlt-tolerant StoragE
Paper title: REFUSE to die 

KaFFee: Karlsruhe Fast Failure rEcovEry
KaFFeIn: Karlsruhe Fast Failure rEcovery In-Memory

## Usage

To use ReStore, first add the repository as a submodule into your project:
```Bash
git submodule add --recursive https://github.com/ReStoreCpp/ReStore.git extern/ReStore
```

Then, include the following into your CMakeLists.txt:
```CMake
add_subdirectory(extern/ReStore)
target_link_libraries("${PROJECT_NAME}" ReStore)
```


