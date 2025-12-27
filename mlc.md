---
layout: default
---

# mlc-llm


## Windows

```
cmake -A x64 -S . -B build 
    -DUSE_METAL=OFF 
    -DUSE_VULKAN=ON 
    -DUSE_CUDA=OFF 
    -DUSE_LLVM={llvm-config.exe}
```

## Intel

```
cmake -S . -B build_amd \
    -DUSE_METAL=ON \
    -DCMAKE_OSX_ARCHITECTURES=x86_64 \
    -DUSE_VULKAN=OFF \
    -DUSE_CUDA=OFF \
    -DCMAKE_C_FLAGS="-O3 -march=haswell" \
    -DCMAKE_CXX_FLAGS="-O3 -march=haswell" \
    -DCMAKE_POLICY_VERSION_MINIMUM="3.5" 
cmake --build build_amd --parallel $(sysctl -n hw.logicalcpu)
```

## Apple Silicon


```
cmake -S . -B build_arm 
    -DUSE_METAL=ON 
    -DUSE_VULKAN=OFF 
    -DUSE_CUDA=OFF
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build_arm --parallel $(sysctl -n hw.logicalcpu)
```
