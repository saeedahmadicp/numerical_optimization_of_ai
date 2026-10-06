/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { catalogPlugin, siteUrlPlugin } from './vite/catalog';

// GitHub Pages: relative base + hash router, so the build works from any sub-path.
export default defineConfig({
  base: './',
  plugins: [react(), catalogPlugin(), siteUrlPlugin()],
  worker: { format: 'es' },
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        // Long-lived vendor chunks: app deploys do not invalidate them.
        manualChunks(id: string) {
          if (id.includes('node_modules/katex')) return 'katex';
          if (
            id.includes('node_modules/motion') ||
            id.includes('node_modules/framer-motion') ||
            id.includes('node_modules/motion-')
          )
            return 'motion';
          if (id.includes('node_modules/react') || id.includes('node_modules/scheduler'))
            return 'react';
          return undefined;
        },
      },
    },
  },
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts', 'src/**/*.test.ts'],
    reporters: ['default'],
  },
});
