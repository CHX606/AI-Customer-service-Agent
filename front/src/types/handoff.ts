export type HandoffStatus = "pending" | "processing" | "resolved";
export type NotificationStatus = "pending" | "sending" | "sent" | "failed" | "unconfigured";

export interface HandoffSubmission {
  tenant_id: string;
  session_id: string;
  submission_id: string;
  issue: string;
}

export interface HandoffReceipt {
  ticket_id: string;
  status: HandoffStatus;
  notification_status: NotificationStatus;
  message: string;
}

export interface HandoffTicket {
  ticket_id: string;
  tenant_id: string;
  session_id: string;
  issue: string;
  status: HandoffStatus;
  notification_status: NotificationStatus;
  notification_error?: string | null;
  created_at: string;
  updated_at?: string;
  handling_note?: string;
}

export interface HandoffDetail extends HandoffTicket {
  handling_note: string;
  messages: { role: string; content: string }[];
}
