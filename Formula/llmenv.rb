class Llmenv < Formula
  desc "C libraries for running large language models locally on Apple Silicon"
  homepage "https://github.com/c-vision/LLMEnv"
  url "https://github.com/c-vision/LLMEnv/releases/download/v0.2.0/llmenv-0.2.0-macos-arm64.tar.gz"
  sha256 "71aa74396ee597e079a5548792c97e986d70bc144e58644ec322a6d47aae4a3b"
  version "0.2.0"
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
