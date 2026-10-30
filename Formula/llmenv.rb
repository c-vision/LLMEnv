class Llmenv < Formula
  desc "C libraries for running large language models locally on Apple Silicon"
  homepage "https://github.com/c-vision/LLMEnv"
  version "0.3.0"
  license "Apache-2.0"

  # Two builds of the same engine: macOS 26.2 and later gets the one with the NAX kernels (M5 tensor units),
  # every other supported macOS (14 and later) gets the compatible one.
  if MacOS.full_version >= "26.2"
    url "https://github.com/c-vision/LLMEnv/releases/download/v0.3.0/llmenv-0.3.0-macos26-arm64.tar.gz"
    sha256 "556fe5cd920a245ca7c0331dd133f07e6375bad947c683c714ed7970a0805a27"
  else
    url "https://github.com/c-vision/LLMEnv/releases/download/v0.3.0/llmenv-0.3.0-macos14-arm64.tar.gz"
    sha256 "692c83ce82291c85631585cc7c945ce43d0fcaa1b0b41118f76cf553f29c0936"
  end

  depends_on arch: :arm64
  depends_on :macos

  def install
    prefix.install "lib", "include", "share"
  end

  test do
    assert_predicate lib/"llmenv/libllm_ai_mlx.dylib", :exist?
    assert_predicate lib/"llmenv/mlx.metallib", :exist?
    assert_predicate lib/"llmenv/LLMENV_VERSION", :exist?
  end
end
