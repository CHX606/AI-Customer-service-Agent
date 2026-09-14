import { useAui } from "@assistant-ui/react";
import { useCallback, useEffect, useRef, useState, type Dispatch, type SetStateAction } from "react";
import { CONVERSATIONS_UPDATED_EVENT, createNewConversation, deleteConversation, getActiveConversationId, getConversations, loadStoredChatMessages, setActiveConversation, type Conversation } from "../lib/chat-storage";

export function useConversations(setIsSidebarOpen: Dispatch<SetStateAction<boolean>>) {
  const aui = useAui();
  const [deleteTarget, setDeleteTarget] = useState<Conversation | null>(null);
  const [activeId, setActiveId] = useState(() => getActiveConversationId());
  const [conversations, setConversations] = useState(() => getConversations());
  const transitionVersionRef = useRef(0);

  useEffect(() => {
    const syncConversations = () => {
      setActiveId(getActiveConversationId());
      setConversations(getConversations());
    };

    window.addEventListener(CONVERSATIONS_UPDATED_EVENT, syncConversations);
    return () => {
      window.removeEventListener(CONVERSATIONS_UPDATED_EVENT, syncConversations);
    };
  }, []);

  const prepareConversationTransition = useCallback(async (version: number) => {
    if (aui.thread.getState().isRunning) {
      aui.thread.cancelRun();

      await new Promise<void>((resolve) => {
        let unsubscribe = () => { };
        const finishWhenIdle = () => {
          if (!aui.thread.getState().isRunning) {
            unsubscribe();
            resolve();
          }
        };

        unsubscribe = aui.subscribe(finishWhenIdle);
        finishWhenIdle();
      });
    }

    if (version !== transitionVersionRef.current) return false;
    await aui.composer.reset();
    return version === transitionVersionRef.current;
  }, [aui]);


  const handleNewConversation = async () => {
    const version = ++transitionVersionRef.current;
    if (!await prepareConversationTransition(version)) return;

    createNewConversation();
    aui.thread.reset([]);
    setIsSidebarOpen(false);
  };

  const handleSwitchConversation = async (id: string) => {
    const version = ++transitionVersionRef.current;
    if (id === getActiveConversationId()) return;

    if (!await prepareConversationTransition(version)) return;

    const messages = loadStoredChatMessages(id);
    if (version !== transitionVersionRef.current) return;

    setActiveConversation(id);
    aui.thread.reset(messages);
    setIsSidebarOpen(false);
  };

  const handleConfirmDelete = async () => {
    if (!deleteTarget) return;
    const version = ++transitionVersionRef.current;
    const isDeletingActive = deleteTarget.id === getActiveConversationId();

    if (isDeletingActive) {
      if (!await prepareConversationTransition(version)) return;
    }

    deleteConversation(deleteTarget.id);

    if (isDeletingActive) {
      const nextId = getActiveConversationId();
      aui.thread.reset(loadStoredChatMessages(nextId));
    }

    setDeleteTarget(null);
  };


  return { activeId, conversations, deleteTarget, setDeleteTarget, handleNewConversation, handleSwitchConversation, handleConfirmDelete };
}
