class Llmenv < Formula
  desc "C libraries for running large language models locally on Apple Silicon"
  homepage "https://github.com/c-vision/LLMEnv"
  url "https://github.com/c-vision/LLMEnv/releases/download/v0.1.0/llmenv-0.1.0-macos-arm64.tar.gz"
  sha256 "719e9b2b5ef5e18e413fb2d108564b75bf5df39ad9872cf64609779139d7e524"
  version "0.1.0"
  license "Apache-2.0"

  depends_on arch: :arm64
  depends_on :macos
  depends_on "openssl@3"

  def install
    prefix.install "lib", "include", "share"
  end

  test do
    assert_predicate lib/"llmenv/libllm_ai_mlx.dylib", :exist?
    assert_predicate lib/"llmenv/mlx.metallib", :exist?
  end
end
