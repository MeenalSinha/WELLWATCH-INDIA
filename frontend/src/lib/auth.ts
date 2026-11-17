import { TokenResponse, UserRole } from '@/types';

const TOKEN_KEY = 'wellwatch_access_token';
const USER_KEY = 'wellwatch_user_profile';

export const getStoredToken = (): string | null => {
  if (typeof window === 'undefined') return null;
  return sessionStorage.getItem(TOKEN_KEY);
};

export const getStoredUser = (): TokenResponse | null => {
  if (typeof window === 'undefined') return null;
  const raw = sessionStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
};

export const setSession = (tokenData: TokenResponse): void => {
  if (typeof window === 'undefined') return;
  sessionStorage.setItem(TOKEN_KEY, tokenData.access_token);
  sessionStorage.setItem(USER_KEY, JSON.stringify(tokenData));
};

export const clearSession = (): void => {
  if (typeof window === 'undefined') return;
  sessionStorage.removeItem(TOKEN_KEY);
  sessionStorage.removeItem(USER_KEY);
};

export const isAuthenticated = (): boolean => {
  return !!getStoredToken();
};

export const hasRole = (allowedRoles: UserRole[]): boolean => {
  const user = getStoredUser();
  if (!user) return false;
  return allowedRoles.includes(user.role);
};
