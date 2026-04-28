import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Produces a self-contained build under .next/standalone – required for
  // the minimal Docker image used in Cloud Run.
  output: "standalone",
};

export default nextConfig;
