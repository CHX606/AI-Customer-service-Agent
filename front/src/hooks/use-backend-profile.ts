import { useCallback, useEffect, useState } from "react";
import { checkBackendHealth } from "../api/chat";
import { DEFAULT_TENANT_ID, fetchPublicProfile } from "../api/profile";
import { DEFAULT_FALLBACK_PROFILE } from "../lib/default-profile";
import type { TenantPublicProfile } from "../types/profile";

export function useBackendProfile() {
  const [isBackendOnline, setIsBackendOnline] = useState<boolean | null>(null);
  const [profile, setProfile] = useState<TenantPublicProfile>(
    DEFAULT_FALLBACK_PROFILE,
  );
  const loadProfile = useCallback(async () => {
    try {
      const data = await fetchPublicProfile(DEFAULT_TENANT_ID);
      setProfile(data);
    } catch {
      // 保持降级配置
    }
  }, []);

  const checkHealth = useCallback(async () => {
    const online = await checkBackendHealth();
    setIsBackendOnline(online);
    if (online) {
      loadProfile();
    }
  }, [loadProfile]);

  useEffect(() => {
    checkHealth();
    const timer = setInterval(checkHealth, 15000);
    return () => clearInterval(timer);
  }, [checkHealth]);


  return { profile, isBackendOnline, loadProfile, checkHealth };
}
