# Originally authored by @jellydn (https://github.com/jellydn/homebrew-tap).
# Maintained in-repo; auto-bumped by .github/workflows/release.yaml on stable releases.
class FffMcp < Formula
  desc "Fast file search toolkit for AI agents (MCP server)"
  homepage "https://github.com/dmtrKovalenko/fff.nvim"
  license "MIT"
  version "0.10.0"

  LIVECHECK_REPO = "dmtrKovalenko/fff.nvim".freeze
  RELEASE_BASE = "https://github.com/dmtrKovalenko/fff.nvim/releases/download".freeze

  on_macos do
    on_arm do
      url "#{RELEASE_BASE}/v#{version}/fff-mcp-aarch64-apple-darwin"
      sha256 "a5b95aa4b5861e5c7440fede6056cc7861223abda5796daabe51f785458211c3"
    end

    on_intel do
      url "#{RELEASE_BASE}/v#{version}/fff-mcp-x86_64-apple-darwin"
      sha256 "81a8acdc7b17b7ff54baaa6a23942974855fdfe4133f97347c9b0cb44e9df79f"
    end
  end

  on_linux do
    on_arm do
      url "#{RELEASE_BASE}/v#{version}/fff-mcp-aarch64-unknown-linux-gnu"
      sha256 "370c3ffcd1be8e560c05eaba2b280a016e313c6061a3f307bae7c7427c4778c7"
    end

    on_intel do
      url "#{RELEASE_BASE}/v#{version}/fff-mcp-x86_64-unknown-linux-gnu"
      sha256 "e252dc1bb0412c2719813ccd0095523676f360dddb0af731a778572ab696b592"
    end
  end

  livecheck do
    url "https://github.com/#{LIVECHECK_REPO}/releases/latest"
    strategy :github_latest
  end

  def install
    if OS.mac?
      if Hardware::CPU.arm?
        bin.install "fff-mcp-aarch64-apple-darwin" => "fff-mcp"
      elsif Hardware::CPU.intel?
        bin.install "fff-mcp-x86_64-apple-darwin" => "fff-mcp"
      end
    elsif OS.linux?
      if Hardware::CPU.arm?
        bin.install "fff-mcp-aarch64-unknown-linux-gnu" => "fff-mcp"
      elsif Hardware::CPU.intel?
        bin.install "fff-mcp-x86_64-unknown-linux-gnu" => "fff-mcp"
      end
    end
  end

  test do
    system bin/"fff-mcp", "--healthcheck"
  end
end
