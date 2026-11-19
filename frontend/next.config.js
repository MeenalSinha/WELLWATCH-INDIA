/** @type {import('next').NextConfig} */
const rawApiUrl = process.env.INTERNAL_BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
let backendBase = rawApiUrl.replace(/\/api\/v1\/?$/, '').replace(/\/+$/, '');
if (!backendBase.startsWith('http://') && !backendBase.startsWith('https://')) {
  backendBase = `https://${backendBase}`;
}

const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${backendBase}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
