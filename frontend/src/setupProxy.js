const { createProxyMiddleware } = require('http-proxy-middleware');
const path = require('path');
const express = require('express');

module.exports = function(app) {
  // Serve static files from the public directory
  // This is a more direct approach than using a proxy for static files
  app.use('/data', express.static(path.join(__dirname, '../public/data')));
  
  // If you still need a proxy for other purposes, you can keep it
  // but with a corrected configuration
  /*
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://localhost:3000',
      changeOrigin: true,
    })
  );
  */
};
