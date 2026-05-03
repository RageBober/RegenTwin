import axios from 'axios';

// В dev-режиме (vite) используем пустой baseURL — запросы идут через vite proxy.
// В Tauri/production — напрямую на бэкенд.
const isTauri = '__TAURI_INTERNALS__' in window || '__TAURI__' in window;
const DEFAULT_BASE = isTauri ? 'http://localhost:8000' : '';
const API_BASE = localStorage.getItem('regentwin-api-url') || DEFAULT_BASE;

export const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const API_V1 = '/api/v1';
export const API_VIZ = '/api/viz';

export function updateApiBaseUrl(url: string) {
  const trimmed = url.trim();
  if (!trimmed) {
    resetApiBaseUrl();
    return;
  }
  apiClient.defaults.baseURL = trimmed;
  localStorage.setItem('regentwin-api-url', trimmed);
}

export function resetApiBaseUrl() {
  localStorage.removeItem('regentwin-api-url');
  apiClient.defaults.baseURL = DEFAULT_BASE;
}
