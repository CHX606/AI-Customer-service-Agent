import { createContext } from "react";
import { DEFAULT_FALLBACK_PROFILE } from "../../lib/default-profile";
import type { TenantPublicProfile } from "../../types/profile";
export { DEFAULT_FALLBACK_PROFILE } from "../../lib/default-profile";
export const ProfileContext = createContext<TenantPublicProfile>(DEFAULT_FALLBACK_PROFILE);
