"use client";
// @ts-nocheck

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
// imports
import { SetStateAction, useEffect, useRef, useState } from "react";
import {
  Search,
  Settings,
  ArrowUp,
  Plus,
  X,
  Menu,
  ThumbsUp,
  ThumbsDown,
  RefreshCw,
  Copy,
  ChevronDown,
  Edit,
  MoreVertical,
  ArrowRight,
  ArrowLeft,
  Check,
  User as UserIcon,
  Mic,
  ChevronRight,
  Paperclip,
  Brain,
  VectorSquare,
  Globe,
  MessageSquare,
  Database,
  HelpCircle,
} from "lucide-react";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

import Image from "next/image";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import ManageProfile from "@/components/ui/mange-profile";
import FormattedContent from "@/components/ui/FormattedContent";

// for authentication
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabaseClient";
import type { User } from "@supabase/supabase-js";
import { v4 as uuidv4 } from "uuid";

// types
interface Message {
  id: string;
  content: string;
  sender: "user" | "assistant" | "ai";
  thinkingTime?: number;
  feedback?: number | null;
  isThinking?: boolean;
  created_at?: string;
  conversation_id?: string;
  images?: Array<{ title: string; url: string }>; // Add images field for displaying retrieved images
  thinkingContent?: string; // Add thinking content field
}

interface BranchItem {
  messages: Message[];
  branchId: string | null; // null for original branch, string for real branches
  isOriginal: boolean;
}

interface HistoryResponse {
  messages: Message[];
  branchesByEditId: Record<
    string,
    Array<{ messages: Message[]; branchId: string | null; isOriginal: boolean }>
  >;
  currentBranchIndexByEditId: Record<string, number>;
}

interface ConversationState {
  messages: Message[];
  originalMessages?: Message[];
  editAtId?: string;
  branchesByEditId?: Record<string, BranchItem[]>;
  currentBranchIndexByEditId?: Record<string, number>;
}

interface LoadHistoryResult extends HistoryResponse {
  activeBranchId: string | null;
  originalMessages: Message[];
}

export default function ChatbotUI() {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  // states for chat/history
  const [history, setHistory] = useState<ConversationState[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [currentBranchId, setCurrentBranchId] = useState<string | null>(null);

  // states for ui/input
  const [inputValue, setInputValue] = useState("");
  const [showLeftSidebar, setShowLeftSidebar] = useState(true);
  const [showRightSidebar, setShowRightSidebar] = useState(true);
  const [openAccordions, setOpenAccordions] = useState<string[]>([]);
  const [isAwaitingResponse, setIsAwaitingResponse] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [mechanisticInterpretability, setMechanisticInterpretability] =
    useState(false);

  // states for enhanced Plus dropdown and active features
  const [showPlusDropdown, setShowPlusDropdown] = useState(false);
  const [activeFeatures, setActiveFeatures] = useState<string[]>([]);
  const [openSubmenu, setOpenSubmenu] = useState<string | null>(null);

  // states for editing
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingText, setEditingText] = useState<string>("");
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  // Engine type toggle state
  const [engineType, setEngineType] = useState<"chat" | "query">("chat");

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  // Get the currently active branch messages
  const getCurrentBranchMessages = () => {
    const conv = history[currentIndex];
    if (!conv) return [];

    console.log("DEBUG: getCurrentBranchMessages - conv:", {
      hasMessages: !!conv.messages,
      messagesCount: conv.messages?.length || 0,
      hasBranches: !!conv.branchesByEditId,
      branchesKeys: Object.keys(conv.branchesByEditId || {}),
      currentBranchId,
    });

    // CRITICAL FIX: First, try to find the correct branch messages based on currentBranchId
    if (
      conv.branchesByEditId &&
      Object.keys(conv.branchesByEditId).length > 0
    ) {
      // Look for the active branch by currentBranchId
      if (currentBranchId) {
        for (const [editId, branches] of Object.entries(
          conv.branchesByEditId
        )) {
          for (const branch of (branches as BranchItem[])) {
            if (branch.branchId === currentBranchId) {
              console.log(
                "DEBUG: getCurrentBranchMessages - Found active branch by ID:",
                {
                  branchId: branch.branchId,
                  messagesCount: branch.messages.length,
                }
              );
              return branch.messages;
            }
          }
        }
      }

      // If currentBranchId is null, look for the original branch
      if (currentBranchId === null) {
        for (const [editId, branches] of Object.entries(
          conv.branchesByEditId
        )) {
          for (const branch of (branches as BranchItem[])) {
            if (branch.isOriginal) {
              console.log(
                "DEBUG: getCurrentBranchMessages - Found original branch:",
                {
                  branchId: branch.branchId,
                  isOriginal: branch.isOriginal,
                  messagesCount: branch.messages.length,
                }
              );
              return branch.messages;
            }
          }
        }

        // CRITICAL FIX: If no original branch found but we have branches, use the first branch
        // This handles the case where we're in the initial state
        if (Object.keys(conv.branchesByEditId).length > 0) {
          const firstEditId = Object.keys(conv.branchesByEditId)[0];
          const firstBranches = conv.branchesByEditId[firstEditId];
          if (firstBranches && firstBranches.length > 0) {
            console.log(
              "DEBUG: getCurrentBranchMessages - Using first available branch as fallback:",
              {
                editId: firstEditId,
                branchId: firstBranches[0].branchId,
                isOriginal: firstBranches[0].isOriginal,
                messagesCount: firstBranches[0].messages.length,
              }
            );
            return firstBranches[0].messages;
          }
        }
      }

      // CRITICAL FIX: If no active branch found, use the current branch index for each edit point
      for (const [editId, branches] of Object.entries(conv.branchesByEditId)) {
        const branchArray = branches as BranchItem[];
        const currentIdx = conv.currentBranchIndexByEditId?.[editId] ?? 0;
        if (branchArray[currentIdx]) {
          console.log(
            "DEBUG: getCurrentBranchMessages - Using current branch index:",
            {
              editId,
              currentIdx,
              branchId: branchArray[currentIdx].branchId,
              messagesCount: branchArray[currentIdx].messages.length,
            }
          );
          return branchArray[currentIdx].messages;
        }
      }

      // Fallback: use the first available branch
      for (const [editId, branches] of Object.entries(conv.branchesByEditId)) {
        const branchArray = branches as BranchItem[];
        if (branchArray.length > 0) {
          console.log(
            "DEBUG: getCurrentBranchMessages - Using first available branch:",
            {
              branchId: branchArray[0].branchId,
              messagesCount: branchArray[0].messages.length,
            }
          );
          return branchArray[0].messages;
        }
      }
    }

    // The main messages field should only be used as a last resort
    // This is updated when switching branches or loading conversations
    if (conv.messages && conv.messages.length > 0) {
      console.log(
        "DEBUG: getCurrentBranchMessages - Using main messages field as fallback:",
        {
          messagesCount: conv.messages.length,
          firstMessage: conv.messages[0]?.content?.substring(0, 50) + "...",
          lastMessage:
            conv.messages[conv.messages.length - 1]?.content?.substring(0, 50) +
            "...",
        }
      );
      return conv.messages;
    }

    console.log("DEBUG: getCurrentBranchMessages - No messages found anywhere");
    return [];
  };

  const currentMessages = getCurrentBranchMessages();
  const isWelcomeState = currentMessages.length === 0;
  const editTextareaRef = useRef<HTMLTextAreaElement | null>(null);

  // rate limiting
  const [showLimitModal, setShowLimitModal] = useState(false);

  // authentication
  const router = useRouter();
  const supabase = createClient();
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // uploading functionality
  const [uploading, setUploading] = useState(false);

  // collapsible thinking
  const [showThoughts, setShowThoughts] = useState<Record<string, boolean>>({});
  const toggleThoughts = (id: string) =>
    setShowThoughts((prev: Record<string, boolean>) => ({ ...prev, [id]: !prev[id] }));

  // thinking step state
  const [currentThinkingStep, setCurrentThinkingStep] = useState<string>("");
  const [thinkingDetails, setThinkingDetails] = useState<string>("");
  const [thinkingLogs, setThinkingLogs] = useState<string[]>([]);
  const [showDetailedThinking, setShowDetailedThinking] = useState(true); // Auto-show by default
  const thinkingLogsEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to latest log
  useEffect(() => {
    if (thinkingLogsEndRef.current && showDetailedThinking && thinkingLogs.length > 0) {
      thinkingLogsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [thinkingLogs, showDetailedThinking]);

  // Helper function to map backend step names to user-friendly messages
  const getThinkingMessage = (step: string, details: string = "") => {
    const stepMessages: Record<string, string> = {
      initializing: "🔧 Initializing...",
      loading_knowledge: "📚 Loading knowledge base...",
      initializing_agent: "🤖 Preparing AI agent...",
      processing_query: "🧠 Processing your question...",
      analyzing_query: "🔍 Analyzing query...",
      retrying: "♻️ Retrying with default settings...",
      formatting_response: "✨ Formatting response...",
      finalizing: "🎯 Finalizing answer...",
      complete: "✅ Response ready!",
      error: "❌ Error occurred",
      log: "📋 Processing..." // For detailed log messages
    };
    
    return stepMessages[step] || "🤔 Thinking...";
  };

  // copy functionality
  const [justCopiedId, setJustCopiedId] = useState<string | null>(null);

  // cancel functionality
  const [abortController, setAbortController] =
    useState<AbortController | null>(null);

  // regenerate functionality
  const [isRegenerating, setIsRegenerating] = useState(false);

  // states for model selection
  const [selectedModel, setSelectedModel] = useState("gemma3:latest");
  const [models, setModels] = useState<string[]>([]);

  // states for RAG and retrieval methods
  const [selectedRagMethod, setSelectedRagMethod] = useState(
    "No Specific RAG Method"
  );
  const [selectedRetrievalMethod, setSelectedRetrievalMethod] =
    useState("local context only");
  const [selectedVectorStore, setSelectedVectorStore] = useState<string | null>(null);
  const [vectorStores, setVectorStores] = useState<{categories: Record<string, string[]>}>({categories: {}});
  const [isDeepResearchDropdownOpen, setIsDeepResearchDropdownOpen] =
    useState(false);
  const [isRetrievalDropdownOpen, setIsRetrievalDropdownOpen] = useState(false);
  const [isVectorStoreDropdownOpen, setIsVectorStoreDropdownOpen] = useState(false);

  // conversations list (left sidebar)
  const [conversations, setConversations] = useState<
    { id: string; title: string }[]
  >([]);

  // right sidebar controls
  const [systemPrompt, setSystemPrompt] = useState(
    "You are a helpful AI assistant for comfort and fitting clothing"
  );
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [specDecoding, setSpecDecoding] = useState(false);
  const [strategy, setStrategy] = useState("no-workflow");
  const [preset, setPreset] = useState("CFIR");

  // manage profile modal
  const [showProfileModal, setShowProfileModal] = useState(false);

  // AI thinking time - removed unused state variable

  const openProfileModal = () => {
    setShowUserMenu(false);
    setShowProfileModal(true);
  };
  const closeProfileModal = () => setShowProfileModal(false);

  // Responsive sidebar behavior
  useEffect(() => {
    const handleResize = () => {
      const isSmallScreen = window.innerWidth < 768; // md breakpoint
      const isMediumScreen = window.innerWidth < 1024; // lg breakpoint

      if (isSmallScreen) {
        // On small screens, close both sidebars
        setShowLeftSidebar(false);
        setShowRightSidebar(false);
      } else if (isMediumScreen) {
        // On medium screens, close right sidebar but keep left sidebar
        setShowRightSidebar(false);
      }
    };

    // Set initial state based on current screen size
    handleResize();

    // Add event listener
    window.addEventListener("resize", handleResize);

    // Cleanup
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // hook for session
  useEffect(() => {
    // fetch current session
    supabase.auth.getSession().then(({ data: { session } }: any) => {
      setUser(session?.user ?? null);
      setLoading(false);
    });

    // listen for future auth changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event: any, session: any) => {
      setUser(session?.user ?? null);
    });

    return () => {
      subscription.unsubscribe();
    };
  }, [router, supabase]);

  // hook for initializing history
  useEffect(() => {
    const initialHistory = conversations.map(() => ({
      messages: [],
      originalMessages: [],
      editAtId: undefined,
      branchesByEditId: {},
      currentBranchIndexByEditId: {},
    }));
    setHistory(initialHistory);
    setCurrentIndex(0);
  }, []);

  // Close Deep Research dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Element;
      if (
        isDeepResearchDropdownOpen &&
        !target.closest("[data-deep-research-container]")
      ) {
        setIsDeepResearchDropdownOpen(false);
      }
      if (
        isRetrievalDropdownOpen &&
        !target.closest("[data-retrieval-container]")
      ) {
        setIsRetrievalDropdownOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isDeepResearchDropdownOpen, isRetrievalDropdownOpen]);

  // Fetch available vector stores
  useEffect(() => {
    async function fetchVectorStores() {
      try {
        const res = await fetch(`${API_URL}/api/vector-stores`, {
          headers: headers,
        });
        
        if (res.ok) {
          const data = await res.json();
          setVectorStores(data);
          
          // Set default vector store (first one from first category)
          const categories = data.categories || {};
          const firstCategory = Object.keys(categories)[0];
          if (firstCategory && categories[firstCategory].length > 0) {
            setSelectedVectorStore(categories[firstCategory][0]);
          }
        }
      } catch (error) {
        console.error("Error fetching vector stores:", error);
      }
    }
    
    fetchVectorStores();
  }, []);

  // hook for auto‐scroll to bottom whenever messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [currentMessages, currentBranchId]);

  // Autosize textbox when editing message
  useEffect(() => {
    if (editTextareaRef.current) {
      const textarea = editTextareaRef.current;
      textarea.style.height = "auto"; // reset
      textarea.style.height = textarea.scrollHeight + "px"; // grow to fit
    }
  }, [editingText, editingId]);

  // load conversations
  useEffect(() => {
    if (loading || typeof window === "undefined") return;

    async function loadConversations() {
      try {
        // get supabase session
        const {
          data: { session },
        } = await supabase.auth.getSession();

        if (session?.access_token) {
          headers.Authorization = `Bearer ${session.access_token}`;
        }

        const res = await fetch(`${API_URL}/api/conversations`, {
          headers,
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          console.error("Fetch error:", res.status, err);
          return;
        }

        const data = await res.json();
        if (!Array.isArray(data)) {
          console.error("Expected an array but got:", data);
          return;
        }

        if (data.length === 0) {
          const newConvo = await handleNewChat();
          if (newConvo) {
            setConversations([newConvo]);
            setHistory([
              {
                messages: [],
                originalMessages: [],
                editAtId: undefined,
                branchesByEditId: {},
                currentBranchIndexByEditId: {},
              },
            ]);
            setCurrentIndex(0);
          }
          return;
        }

        const currentConvId = conversations[currentIndex]?.id;
        setConversations(data);
        const newIndex = currentConvId
          ? data.findIndex((c) => c.id === currentConvId)
          : 0;

        setHistory((prev: ConversationState[]) =>
          data.map((_: any, idx: number) =>
            idx === newIndex && prev[currentIndex]
              ? prev[currentIndex]
              : {
                  messages: [],
                  originalMessages: [],
                  editAtId: undefined,
                  branchesByEditId: {},
                  currentBranchIndexByEditId: {},
                }
          )
        );
        setCurrentIndex(Math.max(0, newIndex));

        // Load the current conversation's history if it exists
        if (data.length > 0 && newIndex >= 0) {
          const currentConversation = data[newIndex];
          console.log(
            "DEBUG: loadConversations - Loading history for current conversation:",
            currentConversation.id
          );
          try {
            const {
              messages,
              originalMessages,
              branchesByEditId = {},
              currentBranchIndexByEditId = {},
              activeBranchId,
            } = await loadHistory(currentConversation.id);

            // Backend already returns thinkingTime correctly, no conversion needed
            const convertedMessages = messages.map((msg: any) => {
              if (msg.sender === "assistant") {
                console.log("DEBUG: Loading AI message from backend:", {
                  id: msg.id,
                  thinkingTime: msg.thinkingTime,
                });
              }
              return msg;
            });

            // Branch messages already have correct field names
            const convertedBranchesByEditId: Record<string, BranchItem[]> = {};
            for (const [editId, branches] of Object.entries(branchesByEditId)) {
              convertedBranchesByEditId[editId] = branches.map(
                (branch: any) => branch
              );
            }

            console.log("DEBUG: loadConversations - Setting history state:", {
              newIndex,
              messagesCount: convertedMessages.length,
              originalMessagesCount: originalMessages.length,
              branchesByEditIdKeys: Object.keys(convertedBranchesByEditId),
            });
            console.log(
              "DEBUG: loadConversations - convertedBranchesByEditId:",
              convertedBranchesByEditId
            );
            console.log(
              "DEBUG: loadConversations - currentBranchIndexByEditId:",
              currentBranchIndexByEditId
            );

            setHistory((prev) => {
              const newHist = [...prev];
              newHist[newIndex] = {
                messages: convertedMessages,
                originalMessages: originalMessages,
                branchesByEditId: convertedBranchesByEditId,
                currentBranchIndexByEditId,
              };
              console.log(
                "DEBUG: loadConversations - Updated history state:",
                newHist[newIndex]
              );
              console.log(
                "DEBUG: loadConversations - branchesByEditId after set:",
                newHist[newIndex].branchesByEditId
              );
              return newHist;
            });

            // Set the current branch ID to match the active branch from backend
            setCurrentBranchId(activeBranchId);
          } catch (err) {
            console.error("Error loading current conversation history:", err);
          }
        }
      } catch (err) {
        console.error("Error loading conversations:", err);
      } finally {
        setLoading(false);
      }
    }

    loadConversations();

    // Debug: Check localStorage for any saved branch information
    console.log(
      "DEBUG: useEffect triggered, checking localStorage for branches"
    );
    const keys = Object.keys(localStorage);
    const branchKeys = keys.filter((key) => key.startsWith("branches_"));
    const indexKeys = keys.filter((key) => key.startsWith("branch_indexes_"));
    console.log("DEBUG: Found localStorage keys:", { branchKeys, indexKeys });

    if (branchKeys.length > 0) {
      branchKeys.forEach((key) => {
        try {
          const branches = JSON.parse(localStorage.getItem(key) || "{}");
          console.log(`DEBUG: localStorage ${key}:`, branches);
        } catch (error) {
          console.error(`DEBUG: Error parsing localStorage ${key}:`, error);
        }
      });
    }
  }, [user, loading]);

  // fetch models list
  useEffect(() => {
    async function loadModels() {
      console.log("DEBUG: Loading models from:", `${API_URL}/api/models`);
      try {
        const res = await fetch(`${API_URL}/api/models`);
        console.log("DEBUG: Models response status:", res.status);

        if (!res.ok) {
          const errorText = await res.text();
          console.error("DEBUG: Models error response:", errorText);
          throw new Error(errorText);
        }

        const data = await res.json();
        console.log("DEBUG: Models response data:", data);
        const { models } = data;
        console.log("DEBUG: Available models:", models);
        setModels(models);
        if (models.length > 0) {
          console.log("DEBUG: Setting initial model to:", models[0]);
          setSelectedModel(models[0]);
        }
      } catch (err) {
        console.error("DEBUG: Error loading models:", err);
      }
    }
    loadModels();
  }, []);

  const handleFeedback = async (messageId: string, value: number) => {
    // check value
    const current = history[currentIndex].messages.find(
      (m) => m.id === messageId
    )?.feedback;
    const newValue = current === value ? null : value;

    // update UI
    setHistory((prev) => {
      const copy = [...prev];
      copy[currentIndex] = {
        ...copy[currentIndex],
        messages: copy[currentIndex].messages.map((m) =>
          m.id === messageId ? { ...m, feedback: newValue } : m
        ),
      };
      return copy;
    });

    try {
      // get supabase session
      const {
        data: { session },
      } = await supabase.auth.getSession();

      if (session?.access_token) {
        headers.Authorization = `Bearer ${session.access_token}`;
      }

      const res = await fetch(`${API_URL}/api/messages/${messageId}/feedback`, {
        method: "POST",
        headers,
        body: JSON.stringify({ rating: newValue }),
      });
    } catch (error) {
      console.error("Could not save feedback:", error);
    }
  };

  // auth helper
  const getAuthHeaders = async () => {
    const {
      data: { session },
    } = await supabase.auth.getSession();
    return {
      "Content-Type": "application/json",
      ...(session?.access_token
        ? { Authorization: `Bearer ${session.access_token}` }
        : {}),
    } as Record<string, string>;
  };

  const handleRegenerate = async (aiMessageId: string) => {
    // Set regenerating state for this specific message
    setIsRegenerating(true);

    setHistory((prev) =>
      prev.map((convo, idx) =>
        idx === currentIndex
          ? {
              ...convo,
              messages: convo.messages.map((m) =>
                m.id === aiMessageId
                  ? {
                      ...m,
                      content: "",
                      isThinking: true,
                      thinkingTime: undefined,
                    }
                  : m
              ),
            }
          : convo
      )
    );

    // Create abort controller for regeneration
    const controller = new AbortController();
    setAbortController(controller);
    setIsAwaitingResponse(true);

    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (session?.access_token) {
        headers.Authorization = `Bearer ${session.access_token}`;
      }

      const msgs = history[currentIndex].messages;
      const idx = msgs.findIndex((m) => m.id === aiMessageId);
      const slice = msgs.slice(0, idx);
      const guestHistory = slice.map((m) => ({
        role: m.sender,
        content: m.content,
      }));

      const res = await fetch(
        `${API_URL}/api/messages/${aiMessageId}/regenerate`,
        {
          method: "POST",
          headers,
          body: JSON.stringify({
            conversation_id: conversations[currentIndex].id,
            history: guestHistory,
            model: selectedModel,
            preset,
            temperature,
            rag_method: selectedRagMethod,
            retrieval_method: selectedRetrievalMethod,
            selected_vector_store: selectedVectorStore,
          }),
          signal: controller.signal,
        }
      );

      if (!res.ok) throw new Error("Regeneration failed");

      // backend returns the updated Message object
      const updatedMsg = await res.json();
      console.log("DEBUG: Regeneration response:", updatedMsg);

      setHistory((prev) => {
        const copy = [...prev];
        // replace the old AI message with the regenerated one
        copy[currentIndex].messages = copy[currentIndex].messages.map((m) =>
          m.id === aiMessageId
            ? {
                id: updatedMsg.id || aiMessageId,
                content: updatedMsg.content,
                sender: "assistant",
                thinkingTime: updatedMsg.thinking_time,
                feedback: updatedMsg.feedback,
                isThinking: false,
                images: updatedMsg.images || [], // Include images in regenerated message
              }
            : m
        );
        return copy;
      });
    } catch (err: any) {
      if (err.name === "AbortError") {
        // Handle cancellation
        setHistory((prev) => {
          const copy = [...prev];
          copy[currentIndex].messages = copy[currentIndex].messages.map((m) =>
            m.id === aiMessageId
              ? {
                  ...m,
                  content: "Regeneration cancelled",
                  isThinking: false,
                  thinkingTime: undefined,
                }
              : m
          );
          return copy;
        });
        return;
      }
      console.error("Regeneration error:", err);

      // Restore original content on error
      setHistory((prev) => {
        const copy = [...prev];
        copy[currentIndex].messages = copy[currentIndex].messages.map((m) =>
          m.id === aiMessageId
            ? {
                ...m,
                content: "Regeneration failed. Please try again.",
                isThinking: false,
                thinkingTime: undefined,
              }
            : m
        );
        return copy;
      });
    } finally {
      setIsRegenerating(false);
      setAbortController(null);
      setIsAwaitingResponse(false);
    }
  };

  const loadHistory = async (convId: string): Promise<LoadHistoryResult> => {
    console.log(
      "DEBUG: loadHistory - Loading history for conversation:",
      convId
    );

    const {
      data: { session },
    } = await supabase.auth.getSession();

    if (session?.access_token) {
      headers.Authorization = `Bearer ${session.access_token}`;
    }

    const res = await fetch(
      `${API_URL}/api/messages/conversation/${convId}/history`,
      {
        method: "GET",
        headers,
      }
    );

    if (!res.ok) {
      console.error(
        "DEBUG: loadHistory - Failed to load history, status:",
        res.status
      );
      throw new Error("Failed to load history");
    }

    const historyData = (await res.json()) as LoadHistoryResult;

    // Helper function to convert image_data to images array
    const convertImageData = (msg: any) => {
      if (msg.image_data) {
        try {
          const imageInfo = typeof msg.image_data === 'string' 
            ? JSON.parse(msg.image_data) 
            : msg.image_data;
          
          if (imageInfo && imageInfo.images && Array.isArray(imageInfo.images)) {
            // NEW FORMAT: Handle multiple images from the images array
            msg.images = imageInfo.images.map((img: any) => {
              // Check if path is already a full URL (e.g., Google Images)
              const isExternalUrl = img.path && (img.path.startsWith('http://') || img.path.startsWith('https://'));
              return {
                title: img.caption || '', // Use caption as title
                url: isExternalUrl 
                  ? img.path  // Use external URL directly
                  : `${API_URL}/api/images/${encodeURIComponent(img.path)}`  // Route local images through backend
              };
            });
            console.log(`DEBUG: Converted ${msg.images.length} images from database image_data`);
          } else if (imageInfo && (imageInfo.name || imageInfo.path)) {
            // LEGACY FORMAT: Handle old single-image format
            const imageName = imageInfo.name || imageInfo.path?.split('\\').pop() || imageInfo.path?.split('/').pop();
            msg.images = [{
              title: '', // Empty title to display only the image without any text
              url: `${API_URL}/api/images/${encodeURIComponent(imageName)}`
            }];
            console.log("DEBUG: Converted 1 image from legacy database format");
          }
        } catch (e) {
          console.error("Failed to parse image_data:", e);
        }
      }
      return msg;
    };

    // Convert "ai" sender back to "assistant" for frontend compatibility and process images
    const convertSender = (msg: any) => {
      let convertedMsg = msg;
      if (msg.sender === "ai") {
        convertedMsg = { ...msg, sender: "assistant" };
      }
      // Convert image_data to images array
      return convertImageData(convertedMsg);
    };

    // Convert all messages
    if (historyData.messages) {
      historyData.messages = historyData.messages.map(convertSender);
    }
    if (historyData.originalMessages) {
      historyData.originalMessages =
        historyData.originalMessages.map(convertSender);
    }
    if (historyData.branchesByEditId) {
      for (const [editId, branches] of Object.entries(
        historyData.branchesByEditId
      )) {
        for (const branch of branches) {
          if (branch.messages) {
            branch.messages = branch.messages.map(convertSender);
          }
        }
      }
    }

    console.log("DEBUG: loadHistory - Received data:", {
      messagesCount: historyData.messages?.length || 0,
      originalMessagesCount: historyData.originalMessages?.length || 0,
      branchesByEditIdKeys: Object.keys(historyData.branchesByEditId || {}),
      activeBranchId: historyData.activeBranchId,
    });

    // CRITICAL FIX: Ensure we have proper branch structure
    if (!historyData.branchesByEditId) {
      historyData.branchesByEditId = {};
    }
    if (!historyData.currentBranchIndexByEditId) {
      historyData.currentBranchIndexByEditId = {};
    }

    // If backend didn't return branches, try to load from localStorage
    if (Object.keys(historyData.branchesByEditId).length === 0) {
      console.log(
        "DEBUG: loadHistory - No branches from backend, checking localStorage"
      );
      try {
        const storedBranches = localStorage.getItem(`branches_${convId}`);
        const storedIndexes = localStorage.getItem(`branch_indexes_${convId}`);

        if (storedBranches) {
          const parsedBranches = JSON.parse(storedBranches);
          console.log(
            "DEBUG: loadHistory - Found branches in localStorage:",
            parsedBranches
          );

          // CRITICAL FIX: Properly merge localStorage branches with backend data
          for (const [editId, branches] of Object.entries(parsedBranches)) {
            if (!historyData.branchesByEditId[editId]) {
              historyData.branchesByEditId[editId] = [];
            }

            // Add localStorage branches that don't exist in backend
            for (const localStorageBranch of branches as any[]) {
              const exists = historyData.branchesByEditId[editId].some(
                (b: any) => b.branchId === localStorageBranch.branchId
              );
              if (!exists) {
                historyData.branchesByEditId[editId].push(localStorageBranch);
              }
            }
          }

          if (storedIndexes) {
            const parsedIndexes = JSON.parse(storedIndexes);
            console.log(
              "DEBUG: loadHistory - Found branch indexes in localStorage:",
              parsedIndexes
            );
            historyData.currentBranchIndexByEditId = {
              ...historyData.currentBranchIndexByEditId,
              ...parsedIndexes,
            };
          }

          console.log(
            "DEBUG: loadHistory - Merged localStorage branches with backend data"
          );
        }
      } catch (error) {
        console.error(
          "DEBUG: loadHistory - Error loading from localStorage:",
          error
        );
      }
    }

    // CRITICAL FIX: Ensure we have an original branch for each edit point
    for (const [editId, branches] of Object.entries(
      historyData.branchesByEditId
    )) {
      const hasOriginal = branches.some((b: any) => b.isOriginal);
      if (!hasOriginal && historyData.originalMessages.length > 0) {
        // Create an original branch if it doesn't exist
        const originalBranch = {
          messages: historyData.originalMessages,
          branchId: null, // null indicates original branch
          isOriginal: true,
        };
        branches.unshift(originalBranch);
        console.log(
          `DEBUG: loadHistory - Created missing original branch for editId ${editId}`
        );
      }

      // CRITICAL FIX: Ensure all branches have proper message structure
      for (const branch of branches) {
        if (branch.messages && Array.isArray(branch.messages)) {
          console.log(
            `DEBUG: loadHistory - Branch ${branch.branchId} (isOriginal: ${branch.isOriginal}) has ${branch.messages.length} messages`
          );
          // Log first few messages to debug
          branch.messages.slice(0, 3).forEach((msg, idx) => {
            console.log(`DEBUG: loadHistory - Message ${idx}:`, {
              id: msg.id,
              sender: msg.sender,
              contentLength: msg.content?.length || 0,
              hasThinkingTime: msg.thinkingTime != null,
            });
          });
        }
      }
    }

    // CRITICAL FIX: Set proper current branch indexes if missing
    for (const [editId, branches] of Object.entries(
      historyData.branchesByEditId
    )) {
      if (!(editId in historyData.currentBranchIndexByEditId)) {
        // Default to the last branch (usually the most recent)
        historyData.currentBranchIndexByEditId[editId] = branches.length - 1;
      }
    }

    console.log("DEBUG: loadHistory - Final branch structure:", {
      branchesByEditId: historyData.branchesByEditId,
      currentBranchIndexByEditId: historyData.currentBranchIndexByEditId,
      activeBranchId: historyData.activeBranchId,
    });

    return historyData;
  };

  // load messages immediately when convo is clicked
  const handleConversationClick = async (idx: number, convId: string) => {
    const {
      messages,
      originalMessages,
      branchesByEditId = {},
      currentBranchIndexByEditId = {},
      activeBranchId,
    } = await loadHistory(convId);

    // Debug logging
    console.log("DEBUG: handleConversationClick - received data:", {
      messages,
      originalMessages,
      branchesByEditId,
      currentBranchIndexByEditId,
      activeBranchId,
    });

    // CRITICAL FIX: Save branch information to localStorage for persistence across refreshes
    if (Object.keys(branchesByEditId).length > 0) {
      localStorage.setItem(
        `branches_${convId}`,
        JSON.stringify(branchesByEditId)
      );
      localStorage.setItem(
        `branch_indexes_${convId}`,
        JSON.stringify(currentBranchIndexByEditId)
      );
      console.log(
        "DEBUG: handleConversationClick - Saved branches to localStorage:",
        branchesByEditId
      );
    }

    // CRITICAL FIX: Ensure we have proper branch structure
    const processedBranchesByEditId = { ...branchesByEditId };
    const processedCurrentBranchIndexByEditId = {
      ...currentBranchIndexByEditId,
    };

    // Ensure each edit point has an original branch
    for (const [editId, branches] of Object.entries(
      processedBranchesByEditId
    )) {
      const hasOriginal = branches.some((b: any) => b.isOriginal);
      if (!hasOriginal && originalMessages.length > 0) {
        // Create an original branch if it doesn't exist
        const originalBranch = {
          messages: originalMessages,
          branchId: null, // null indicates original branch
          isOriginal: true,
        };
        branches.unshift(originalBranch);
        console.log(
          `DEBUG: handleConversationClick - Created missing original branch for editId ${editId}`
        );
      }

      // Set proper current branch index if missing
      if (!(editId in processedCurrentBranchIndexByEditId)) {
        processedCurrentBranchIndexByEditId[editId] = branches.length - 1;
      }
    }

    console.log(
      "DEBUG: handleConversationClick - Processed branch structure:",
      {
        branchesByEditId: processedBranchesByEditId,
        currentBranchIndexByEditId: processedCurrentBranchIndexByEditId,
      }
    );

    setHistory((prev) => {
      const newHist = [...prev];
      newHist[idx] = {
        messages: messages, // This contains the active branch messages from backend
        originalMessages: originalMessages,
        branchesByEditId: processedBranchesByEditId,
        currentBranchIndexByEditId: processedCurrentBranchIndexByEditId,
      };
      return newHist;
    });

    setCurrentIndex(idx);
    setCurrentBranchId(activeBranchId);

    console.log("DEBUG: handleConversationClick - Updated history state:", {
      messagesCount: messages.length,
      branchesByEditIdKeys: Object.keys(processedBranchesByEditId),
      currentBranchId: activeBranchId,
    });
  };

  // move conversation history
  const moveConversationToTop = (idx: number) => {
    setConversations((prev) => {
      const arr = [...prev];
      const [chosenConv] = arr.splice(idx, 1);
      return [chosenConv, ...arr];
    });

    setHistory((prev) => {
      const arr = [...prev];
      const [chosenHist] = arr.splice(idx, 1);
      return [chosenHist, ...arr];
    });

    setCurrentIndex(0);
  };

  // generate AI response using the selected model
  const generateAIResponse = async (messages: any[], model: string) => {
    console.log("DEBUG: generateAIResponse called");
    console.log("DEBUG: API_URL:", API_URL);

    const {
      data: { session },
    } = await supabase.auth.getSession();

    const payload = {
      conversation_id: conversations[currentIndex].id,
      messages: messages.map((m) => ({
        id: m.id,
        conversation_id: conversations[currentIndex].id,
        sender: m.sender === "assistant" ? "ai" : m.sender, // Convert "assistant" to "ai" for database
        content: m.content,
        thinking_time: m.thinkingTime,
        feedback: m.feedback,
        model: selectedModel,
        preset: preset,
        system_prompt: systemPrompt,
        speculative_decoding: specDecoding,
        temperature: temperature,
        top_p: topP,
        strategy: strategy,
        rag_method: selectedRagMethod,
        retrieval_method: selectedRetrievalMethod,
        selected_vector_store: selectedVectorStore,
      })),
      system_prompt: systemPrompt,
      model: selectedModel,
      temperature,
      top_p: topP,
      speculative_decoding: specDecoding,
      strategy,
      preset,
      rag_method: selectedRagMethod,
      retrieval_method: selectedRetrievalMethod,
      selected_vector_store: selectedVectorStore,
    };

    console.log("DEBUG: Request payload:", payload);
    console.log("DEBUG: Request URL:", `${API_URL}/api/chat`);
    console.log("DEBUG: Selected model in generateAIResponse:", selectedModel);
    console.log("DEBUG: Model parameter passed to generateAIResponse:", model);

    if (session?.access_token) {
      headers.Authorization = `Bearer ${session.access_token}`;
      console.log("DEBUG: Auth token present");
    } else {
      console.log("DEBUG: No auth token");
    }

    console.log("DEBUG: Request headers:", headers);

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });

      console.log("DEBUG: Response status:", response.status);
      console.log(
        "DEBUG: Response headers:",
        Object.fromEntries(response.headers.entries())
      );

      if (!response.ok) {
        const errorText = await response.text();
        console.error("DEBUG: Response error text:", errorText);
        console.error("DEBUG: Failed payload:", payload);
        throw new Error(
          `⚠️ Failed to generate AI response. Status: ${response.status}, Error: ${errorText}`
        );
      }

      const responseData = await response.json();
      console.log("DEBUG: Response data:", responseData);
      return responseData;
    } catch (error) {
      console.error("DEBUG: Fetch error:", error);
      throw error;
    }
  };

  var limitReached = false;
  const hasMultipleBranches = (messageId: string): boolean => {
    const conv = history[currentIndex];
    if (!conv) return false;

    // Check if this message is an edit point (has branches)
    if (conv.branchesByEditId && conv.branchesByEditId[messageId]) {
      const branches = conv.branchesByEditId[messageId];
      console.log(
        `DEBUG: hasMultipleBranches - message ${messageId} is an edit point with ${branches.length} branches:`,
        branches
      );
      // CRITICAL FIX: Show navigation if there are multiple branches OR if there's an original + at least one branch
      const hasOriginal = branches.some((b: any) => b.isOriginal);
      const hasBranches = branches.some((b: any) => !b.isOriginal);
      return branches.length > 1 || (hasOriginal && hasBranches);
    }

    // CRITICAL FIX: Add debugging for when branches exist but not for this message
    if (
      conv.branchesByEditId &&
      Object.keys(conv.branchesByEditId).length > 0
    ) {
      console.log(
        `DEBUG: hasMultipleBranches - message ${messageId} not found in branches, but branches exist:`,
        {
          messageId,
          existingEditIds: Object.keys(conv.branchesByEditId),
          branchesByEditId: conv.branchesByEditId,
        }
      );
    }

    // Only show navigation for edit points, not for messages within branches
    console.log(
      `DEBUG: hasMultipleBranches - message ${messageId} is not an edit point, no navigation needed`
    );
    return false;
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isAwaitingResponse) return;

    const {
      data: { session },
    } = await supabase.auth.getSession();
    let conversationId: string;

    if (conversations.length === 0) {
      const newConversation = await handleNewChat();
      if (!newConversation) return;
      conversationId = newConversation.id;
    } else {
      conversationId = conversations[currentIndex].id;
    }

    const userMessage: Message = {
      id: uuidv4(),
      content: inputValue,
      sender: "user",
      thinkingTime: 0, // User messages have no thinking time
    };

    setHistory((prev) => {
      const copy = [...prev];
      copy[currentIndex] = {
        ...copy[currentIndex],
        messages: [...copy[currentIndex].messages, userMessage],
        editAtId: undefined,
      };
      return copy;
    });
    setInputValue("");
    const controller = new AbortController();
    setAbortController(controller);
    setIsAwaitingResponse(true);

    // Check if we're on a branch and if it's the original branch
    let isOriginalBranch = false;
    if (currentBranchId && history[currentIndex].branchesByEditId) {
      for (const [editId, branches] of Object.entries(
        history[currentIndex].branchesByEditId
      )) {
        const branch = (branches as any[]).find((b: any) => b.branchId === currentBranchId);
        if (branch && branch.isOriginal) {
          isOriginalBranch = true;
          break;
        }
      }
    }

    // if its a branch (but not the original branch) and has a valid branchId
    if (
      currentBranchId &&
      currentBranchId !== "original" &&
      !isOriginalBranch
    ) {
      console.log(
        "DEBUG: handleSendMessage - Processing branch message on branch:",
        currentBranchId
      );

      // CRITICAL FIX: Verify we're on the correct branch before proceeding
      const conv = history[currentIndex];
      let targetBranchId = currentBranchId;

      // Double-check that currentBranchId matches an actual branch
      if (conv.branchesByEditId) {
        let foundBranch = false;
        for (const [editId, branches] of Object.entries(
          conv.branchesByEditId
        )) {
          for (const branch of (branches as any[])) {
            if (branch.branchId === currentBranchId) {
              foundBranch = true;
              console.log(
                "DEBUG: handleSendMessage - Confirmed branch exists:",
                {
                  editId,
                  branchId: currentBranchId,
                  isOriginal: branch.isOriginal,
                }
              );
              break;
            }
          }
          if (foundBranch) break;
        }

        if (!foundBranch) {
          console.error(
            "DEBUG: handleSendMessage - currentBranchId not found in branches, this will cause wrong branch targeting!"
          );
        }
      }

      const messagesForAI = [
        ...(history[currentIndex]?.messages || []),
        userMessage,
      ];
      const responseData = await generateAIResponse(messagesForAI, selectedModel);
      const { result: aiResp, duration, image_info } = responseData;

      // Convert image_info to the images array format expected by the Message interface
      let images: Array<{ title: string; url: string }> = [];
      if (image_info && image_info.images && Array.isArray(image_info.images)) {
        // NEW: Handle multiple images from the images array
        images = image_info.images.map((img: any) => {
          // Check if path is already a full URL (e.g., Google Images)
          const isExternalUrl = img.path && (img.path.startsWith('http://') || img.path.startsWith('https://'));
          return {
            title: img.caption || '', // Use caption as title
            url: isExternalUrl 
              ? img.path  // Use external URL directly
              : `${API_URL}/api/images/${encodeURIComponent(img.path)}`  // Route local images through backend
          };
        });
        console.log(`DEBUG: Loaded ${images.length} images for branch message`);
      } else if (image_info && (image_info.name || image_info.path)) {
        // LEGACY: Handle old single-image format (backward compatibility)
        const imageName = image_info.name || image_info.path?.split('\\').pop() || image_info.path?.split('/').pop();
        images = [{
          title: '', // Empty title to display only the image without any text
          url: `${API_URL}/api/images/${encodeURIComponent(imageName)}`
        }];
        console.log("DEBUG: Loaded 1 image from legacy format for branch");
      }

      const newAi: Message = {
        id: uuidv4(),
        content: aiResp,
        sender: "assistant",
        thinkingTime: duration,
        images: images, // Include images in the new AI message
      };

      const updatedBranchMessages = [...messagesForAI, newAi];
      const backendMessages = updatedBranchMessages.map(toBackendMessage);

      if (session?.access_token) {
        headers.Authorization = `Bearer ${session.access_token}`;
      }

      // CRITICAL FIX: Save the updated branch messages to the backend
      try {
        // CRITICAL VERIFICATION: Double-check we're using the right branch ID
        console.log("DEBUG: handleSendMessage - VERIFICATION:", {
          currentBranchId,
          targetBranchId,
          areEqual: currentBranchId === targetBranchId,
          messagesCount: backendMessages.length,
        });

        // CRITICAL FIX: Only update branch if we have a valid branch ID
        if (
          targetBranchId &&
          targetBranchId !== "undefined" &&
          targetBranchId !== "null"
        ) {
          console.log(
            "DEBUG: handleSendMessage - Sending branch update to backend:",
            {
              branchId: targetBranchId,
              messagesCount: backendMessages.length,
              messages: backendMessages.map((m) => ({
                id: m.id,
                sender: m.sender,
                contentLength: m.content?.length || 0,
              })),
            }
          );

          const branchUpdateRes = await fetch(
            `${API_URL}/api/branches/${targetBranchId}`,
            {
              method: "PATCH",
              headers,
              body: JSON.stringify(backendMessages),
            }
          );

          if (!branchUpdateRes.ok) {
            const errorText = await branchUpdateRes.text();
            console.error(
              "Failed to update branch in backend:",
              branchUpdateRes.status,
              errorText
            );
            throw new Error(
              `Failed to update branch: ${branchUpdateRes.status} - ${errorText}`
            );
          }

          const updateResult = await branchUpdateRes.json();
          console.log(
            "DEBUG: handleSendMessage - Branch updated successfully in backend:",
            updateResult
          );
        } else {
          console.warn(
            "DEBUG: handleSendMessage - Skipping branch update, invalid branch ID:",
            targetBranchId
          );
        }
      } catch (error) {
        console.error("Error updating branch:", error);
        // Continue with local update even if backend fails
      }

      setHistory((prev) => {
        const copy = [...prev];
        const conv = copy[currentIndex];
        let found = false;
        let editId = null;
        let branchIdx = null;
        if (conv.branchesByEditId) {
          for (const [eid, branches] of Object.entries(conv.branchesByEditId)) {
            const branchArray = branches as BranchItem[];
            const idx = branchArray.findIndex(
              (b: BranchItem) => b.branchId === currentBranchId
            );
            if (idx !== -1) {
              editId = eid;
              branchIdx = idx;
              found = true;
              break;
            }
          }
        }
        if (found && editId !== null && branchIdx !== null) {
          const updatedBranches = [...(conv.branchesByEditId?.[editId] || [])];
          updatedBranches[branchIdx] = {
            ...updatedBranches[branchIdx],
            messages: updatedBranchMessages,
          };
          copy[currentIndex] = {
            ...conv,
            // Update both the branch and the main messages field to keep them in sync
            messages: updatedBranchMessages,
            branchesByEditId: {
              ...conv.branchesByEditId,
              [editId]: updatedBranches,
            },
          };

          // CRITICAL: Also update localStorage to persist the updated branch
          try {
            const currentConversation = conversations[currentIndex];
            if (currentConversation) {
              localStorage.setItem(
                `branches_${currentConversation.id}`,
                JSON.stringify(copy[currentIndex].branchesByEditId)
              );
              console.log(
                "DEBUG: handleSendMessage - Saved updated branches to localStorage"
              );
            }
          } catch (error) {
            console.error(
              "DEBUG: handleSendMessage - Error saving to localStorage:",
              error
            );
          }
        } else {
          // If not in a branch, update the main messages
          copy[currentIndex] = {
            ...conv,
            messages: updatedBranchMessages,
          };
        }
        return copy;
      });

      setIsAwaitingResponse(false);
      return;
    }

    // normal flow - when not in a branch or when on the original branch
    const messagesForAI = [
      ...(history[currentIndex]?.messages || []),
      userMessage,
    ];

    const payload = {
      conversation_id: conversationId,
      messages: [
        ...(history[currentIndex]?.messages || []).map((m) => ({
          id: m.id,
          conversation_id: conversationId,
          sender: m.sender,
          content: m.content,
          thinking_time: m.thinkingTime,
          feedback: m.feedback,
          model: selectedModel,
          preset,
          system_prompt: systemPrompt,
          speculative_decoding: specDecoding,
          temperature,
          top_p: topP,
          strategy,
          rag_method: selectedRagMethod,
          retrieval_method: selectedRetrievalMethod,
          selected_vector_store: selectedVectorStore,
        })),
        {
          id: userMessage.id,
          conversation_id: conversationId,
          sender: "user",
          content: userMessage.content,
          thinking_time: 0, // User messages have no thinking time
        },
      ],
      system_prompt: systemPrompt,
      model: selectedModel,
      temperature,
      top_p: topP,
      speculative_decoding: specDecoding,
      strategy,
      preset,
      rag_method: selectedRagMethod,
      retrieval_method: selectedRetrievalMethod,
      selected_vector_store: selectedVectorStore,
    };

    if (session?.access_token) {
      headers.Authorization = `Bearer ${session.access_token}`;
    }
    try {
      console.log("DEBUG: Sending chat request to:", `${API_URL}/api/chat/stream`);
      console.log("DEBUG: Chat payload:", payload);
      console.log("DEBUG: Selected model in payload:", payload.model);
      console.log("DEBUG: Selected model state:", selectedModel);

      // Reset thinking state
      setCurrentThinkingStep("Initializing...");
      setThinkingDetails("");
      setThinkingLogs([]);
      setShowDetailedThinking(true); // Auto-show logs

      const res = await fetch(`${API_URL}/api/chat/stream`, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      console.log("DEBUG: Chat response status:", res.status);
      console.log(
        "DEBUG: Chat response headers:",
        Object.fromEntries(res.headers.entries())
      );

      // rate limit pop up
      if (res.status === 429) {
        console.log("DEBUG: Rate limit reached");
        limitReached = true;
        setShowLimitModal(true);
        return;
      }

      if (!res.ok) {
        const errorText = await res.text();
        console.error("DEBUG: Chat API error response:", errorText);
        throw new Error(`Chat API error: ${res.status} - ${errorText}`);
      }

      // Handle streaming response
      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      
      if (!reader) {
        throw new Error("No reader available from response");
      }

      let buffer = "";
      let finalData: any = null;

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        
        // Process complete SSE messages
        const lines = buffer.split('\n');
        buffer = lines.pop() || ""; // Keep incomplete line in buffer
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const jsonData = JSON.parse(line.slice(6));
              
              if (jsonData.type === 'progress') {
                // Update thinking step display
                const friendlyMessage = getThinkingMessage(jsonData.step, jsonData.details);
                setCurrentThinkingStep(friendlyMessage);
                setThinkingDetails(jsonData.details || "");
                
                // Add to logs if it's a log message
                if (jsonData.step === 'log') {
                  setThinkingLogs(prev => [...prev, jsonData.details]);
                }
                
                console.log("DEBUG: Progress update:", friendlyMessage, jsonData.details);
              } else if (jsonData.type === 'complete') {
                // Store final data
                finalData = jsonData;
                console.log("DEBUG: Received complete data:", finalData);
              } else if (jsonData.type === 'error') {
                throw new Error(jsonData.message);
              }
            } catch (e) {
              console.error("Error parsing SSE data:", e, line);
            }
          }
        }
      }

      // Reset thinking state
      setCurrentThinkingStep("");
      setThinkingDetails("");
      setThinkingLogs([]);

      if (!finalData) {
        throw new Error("No final data received from stream");
      }

      const { content: aiText, duration, ai_message, image_info } = finalData;
      console.log("DEBUG: AI message from response:", ai_message);
      console.log("DEBUG: Image info from response:", image_info);
      console.log(
        "DEBUG: AI message thinking_time:",
        ai_message?.thinking_time
      );
      console.log("DEBUG: Duration from response:", duration);
      
      // Convert image_info to the images array format expected by the Message interface
      let images: Array<{ title: string; url: string }> = [];
      if (image_info && image_info.images && Array.isArray(image_info.images)) {
        // NEW: Handle multiple images from the images array
        images = image_info.images.map((img: any) => {
          // Check if path is already a full URL (e.g., Google Images)
          const isExternalUrl = img.path && (img.path.startsWith('http://') || img.path.startsWith('https://'));
          return {
            title: img.caption || '', // Use caption as title
            url: isExternalUrl 
              ? img.path  // Use external URL directly
              : `${API_URL}/api/images/${encodeURIComponent(img.path)}`  // Route local images through backend
          };
        });
        console.log(`DEBUG: Loaded ${images.length} images from image_info`);
      } else if (image_info && (image_info.name || image_info.path)) {
        // LEGACY: Handle old single-image format (backward compatibility)
        const imageName = image_info.name || image_info.path?.split('\\').pop() || image_info.path?.split('/').pop();
        images = [{
          title: '', // Empty title to display only the image without any text
          url: `${API_URL}/api/images/${encodeURIComponent(imageName)}`
        }];
        console.log("DEBUG: Loaded 1 image from legacy format");
      } else {
        console.log("DEBUG: No images found in image_info");
      }
      
      const newMessage = {
        id: ai_message.id,
        content: ai_message.content,
        sender: ai_message.sender === "ai" ? "assistant" : ai_message.sender, // Convert "ai" back to "assistant" for frontend
        thinkingTime: ai_message.thinking_time || duration || 0, // Ensure this is never null
        images: images, // Include images in the message
        thinkingContent: ai_message.thinking_content || finalData.thinking_content, // Include thinking content
      };
      console.log("DEBUG: New message being added:", newMessage);

      setHistory((prev) => {
        const copy = [...prev];
        const updatedMessages = [...copy[currentIndex].messages, newMessage];

        // Always update the main messages field first
        copy[currentIndex] = {
          ...copy[currentIndex],
          messages: updatedMessages,
          editAtId: undefined,
        };

        // CRITICAL FIX: Always update branch messages when on any branch (but not original with null branchId)
        if (
          currentBranchId &&
          currentBranchId !== "original" &&
          copy[currentIndex].branchesByEditId
        ) {
          let found = false;
          let editId = null;
          let branchIdx = null;

          for (const [eid, branches] of Object.entries(
            copy[currentIndex].branchesByEditId
          )) {
            const branchArray = branches as BranchItem[];
            const idx = branchArray.findIndex(
              (b: BranchItem) => b.branchId === currentBranchId
            );
            if (idx !== -1) {
              editId = eid;
              branchIdx = idx;
              found = true;
              break;
            }
          }

          if (found && editId !== null && branchIdx !== null) {
            const updatedBranches = [
              ...(copy[currentIndex].branchesByEditId?.[editId] || []),
            ];
            updatedBranches[branchIdx] = {
              ...updatedBranches[branchIdx],
              messages: updatedMessages,
            };
            copy[currentIndex].branchesByEditId[editId] = updatedBranches;

            console.log("DEBUG: handleSendMessage - Updated branch messages:", {
              editId,
              branchIdx,
              branchId: currentBranchId,
              messagesCount: updatedMessages.length,
              isOriginal: updatedBranches[branchIdx].isOriginal,
              branchMessagesCount: updatedBranches[branchIdx].messages.length,
            });

            // CRITICAL: Also update localStorage to persist the updated branch
            try {
              const currentConversation = conversations[currentIndex];
              if (currentConversation) {
                localStorage.setItem(
                  `branches_${currentConversation.id}`,
                  JSON.stringify(copy[currentIndex].branchesByEditId)
                );
                console.log(
                  "DEBUG: handleSendMessage - Saved updated branches to localStorage"
                );
              }
            } catch (error) {
              console.error(
                "DEBUG: handleSendMessage - Error saving to localStorage:",
                error
              );
            }
          }
        }

        console.log(
          "DEBUG: Updated history with new message:",
          copy[currentIndex].messages
        );
        console.log(
          "DEBUG: Last message thinking time:",
          copy[currentIndex].messages[copy[currentIndex].messages.length - 1]
            ?.thinkingTime
        );
        return copy;
      });

      setIsAwaitingResponse(false);

      // title generation for first message
      const wasFirst = history[currentIndex]?.messages.length === 0;
      if (session?.access_token) {
        headers.Authorization = `Bearer ${session.access_token}`;
      }
      if (wasFirst) {
        const titleRes = await fetch(`${API_URL}/api/title`, {
          method: "POST",
          headers,
          body: JSON.stringify({
            conversation_id: conversations[currentIndex].id,
            user_message: userMessage.content,
            ai_response: aiText,
          }),
        });
        const { title: finalTitle } = await titleRes.json();
        // Update local list
        setConversations((prev) =>
          prev.map((c, i) =>
            i === currentIndex ? { ...c, title: finalTitle } : c
          )
        );

        // Persist the new title
        if (session?.access_token) {
          headers.Authorization = `Bearer ${session.access_token}`;
        }
        await fetch(`${API_URL}/api/conversations/${conversationId}`, {
          method: "PATCH",
          headers,
          body: JSON.stringify({ title: finalTitle }),
        });
      }

      moveConversationToTop(currentIndex);
    } catch (err: any) {
      // Reset thinking state on error
      setCurrentThinkingStep("");
      setThinkingDetails("");
      setThinkingLogs([]);
      
      if (err.name === "AbortError") {
        return;
      }
      console.error("Error in handleSendMessage:", err);
    } finally {
      setAbortController(null);
      setIsAwaitingResponse(false);
      // Ensure thinking state is cleared
      setCurrentThinkingStep("");
      setThinkingDetails("");
      setThinkingLogs([]);
    }
  };

  // cancel current AI request
  const handleCancel = () => {
    // Reset thinking state on cancel
    setCurrentThinkingStep("");
    setThinkingDetails("");
    setThinkingLogs([]);
    
    if (abortController) {
      abortController.abort();
      setAbortController(null);
      const cancelMsg: Message = {
        id: uuidv4(),
        content: "Message cancelled",
        sender: "assistant",
      };
      setHistory((prev) => {
        const copy = [...prev];
        const conv = copy[currentIndex];
        copy[currentIndex] = {
          ...conv,
          messages: [...conv.messages, cancelMsg],
        };
        return copy;
      });
    }
    setIsAwaitingResponse(false);
  };

  // Edit message
  const startEdit = (messageId: string, currentContent: string) => {
    console.log("DEBUG: startEdit - Starting edit for message:", {
      messageId,
      currentContentLength: currentContent.length,
      currentIndex,
      hasHistory: !!history[currentIndex],
      messagesCount: history[currentIndex]?.messages?.length || 0,
      originalMessagesCount:
        history[currentIndex]?.originalMessages?.length || 0,
    });

    // Validate the message exists before starting edit
    const conv = history[currentIndex];
    if (conv) {
      const messageExists =
        conv.messages?.some((m) => m.id === messageId) ||
        conv.originalMessages?.some((m) => m.id === messageId);

      if (!messageExists) {
        console.warn("DEBUG: startEdit - Message not found in current state:", {
          messageId,
          availableIds: {
            messages: conv.messages?.map((m) => m.id) || [],
            originalMessages: conv.originalMessages?.map((m) => m.id) || [],
          },
        });

        // Message not found - this could cause issues when trying to commit the edit
        console.warn("DEBUG: startEdit - Message not found, edit may fail");
        // Continue anyway as the user might want to try
      }
    }

    // CRITICAL FIX: Clear any existing edit state to prevent conflicts
    setEditingId(null);
    setEditingText("");

    // CRITICAL FIX: Use setTimeout to ensure state is cleared before setting new values
    setTimeout(() => {
      setEditingId(messageId);
      setEditingText(currentContent);
    }, 0);
  };

  // Cancel editing
  const cancelEdit = () => {
    setEditingId(null);
    setEditingText("");
  };

  const getBranchIndex = (messageId: string): number => {
    const conv = history[currentIndex];
    return conv.currentBranchIndexByEditId?.[messageId] ?? 0;
  };

  function toBackendMessage(msg: Message) {
    return {
      ...msg,
      thinking_time: msg.thinkingTime,
      feedback: msg.feedback ?? null,
      thinkingTime: undefined,
    };
  }
  // branch logic
  const commitEdit = async (messageId: string) => {
    // CRITICAL FIX: Prevent multiple simultaneous edits
    if (isAwaitingResponse) {
      console.warn(
        "DEBUG: commitEdit - Already processing edit, ignoring duplicate call"
      );
      return;
    }

    const trimmed = editingText.trim();
    if (!trimmed) return;

    const originalMessages =
      history[currentIndex].originalMessages || history[currentIndex].messages;
    const msgIdx = originalMessages.findIndex((m) => m.id === messageId);
    if (msgIdx === -1) return;
    const original = originalMessages[msgIdx].content;
    const hasChanged = trimmed !== original;

    setEditingId(null);
    setEditingText("");

    // create a new branch with the edited content
    const updatedOriginalMsgs = [...originalMessages];
    updatedOriginalMsgs[msgIdx] = {
      ...updatedOriginalMsgs[msgIdx],
      content: trimmed,
    };

    // Set loading state before making API calls
    setIsAwaitingResponse(true);

    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();

      // Set authorization header for API calls
      if (session?.access_token) {
        headers.Authorization = `Bearer ${session.access_token}`;
      }

      // create a branch when editing
      const messagesUpToEdit = updatedOriginalMsgs.slice(0, msgIdx + 1);

      // get AI reply
      const {
        result: aiResp,
        duration,
        images,
      } = await generateAIResponse(messagesUpToEdit, selectedModel);
      const newAi: Message = {
        id: Date.now().toString(),
        content: aiResp,
        sender: "assistant",
        thinkingTime: duration,
        images: images || [], // Include images in the new AI message
      };

      const branchedMessages = [...messagesUpToEdit, newAi];
      const convId = conversations[currentIndex].id;

      // Create branch using backend API
      const branchResponse = await fetch(
        `${API_URL}/api/messages/conversations/${convId}/branches`,
        {
          method: "POST",
          headers,
          body: JSON.stringify({
            edit_at_id: messageId,
            messages: branchedMessages,
          }),
        }
      );

      if (!branchResponse.ok) {
        throw new Error(
          `Failed to create branch: ${branchResponse.statusText}`
        );
      }

      const branchData = await branchResponse.json();
      const newBranchId = branchData.branch_id;

      // update state with proper branch structure
      setHistory((prev) => {
        const newHist = [...prev];
        const conv = newHist[currentIndex];
        const eid = messageId;

        // grab existing branches for this edit, default empty
        const existing = conv.branchesByEditId?.[eid] ?? [];

        const updatedList: BranchItem[] = existing.length
          ? [
              ...existing,
              {
                messages: branchedMessages,
                branchId: newBranchId,
                isOriginal: false,
              },
            ]
          : [
              { messages: originalMessages, branchId: null, isOriginal: true },
              {
                messages: branchedMessages,
                branchId: newBranchId,
                isOriginal: false,
              },
            ];

        newHist[currentIndex] = {
          ...conv,
          // CRITICAL FIX: Set messages to the new branch messages
          messages: branchedMessages,
          branchesByEditId: {
            ...(conv.branchesByEditId || {}),
            [eid]: updatedList,
          },
          currentBranchIndexByEditId: {
            ...(conv.currentBranchIndexByEditId || {}),
            [eid]: updatedList.length - 1,
          },
          originalMessages: originalMessages, // Keep original messages unchanged
        };
        return newHist;
      });

      // set current branch ID to the new branch
      setCurrentBranchId(newBranchId);

      // save to localStorage for persistence
      try {
        const currentConversation = conversations[currentIndex];
        if (currentConversation) {
          const updatedHistory = history[currentIndex];
          localStorage.setItem(
            `branches_${currentConversation.id}`,
            JSON.stringify(updatedHistory.branchesByEditId)
          );
          localStorage.setItem(
            `branch_indexes_${currentConversation.id}`,
            JSON.stringify(updatedHistory.currentBranchIndexByEditId)
          );
          console.log(
            "DEBUG: commitEdit - Saved new branch to localStorage:",
            newBranchId
          );
        }
      } catch (error) {
        console.error(
          "DEBUG: commitEdit - Error saving to localStorage:",
          error
        );
      }
    } catch (error) {
      console.error("Error creating branch:", error);
      // restore original state on error
      setHistory((prev) => {
        const copy = [...prev];
        copy[currentIndex] = {
          ...copy[currentIndex],
          messages: originalMessages, // Restore original messages
        };
        return copy;
      });
    } finally {
      setIsAwaitingResponse(false);
    }
  };

  const activateBranch = async (branchId: string | null) => {
    if (!branchId) return;
    const {
      data: { session },
    } = await supabase.auth.getSession();
    if (session?.access_token) {
      headers.Authorization = `Bearer ${session.access_token}`;
    }
    await fetch(`${API_URL}/api/branches/${branchId}/activate`, {
      method: "POST",
      headers,
    });
  };

  const goToPrev = async (messageId: string) => {
    const conv = history[currentIndex];
    const branches = conv.branchesByEditId?.[messageId] || [];
    const currentIdx = conv.currentBranchIndexByEditId?.[messageId] || 0;

    if (branches.length <= 1) return;
    const prevIdx = (currentIdx - 1 + branches.length) % branches.length;
    const selectedBranch = branches[prevIdx];
    // for debug
    await activateBranch(selectedBranch.branchId);
    console.log("selectedBranch.messages", selectedBranch.messages);
    setHistory((prev) => {
      const newHist = [...prev];
      newHist[currentIndex] = {
        ...conv,
        messages: selectedBranch.messages,
        currentBranchIndexByEditId: {
          ...(conv.currentBranchIndexByEditId || {}),
          [messageId]: prevIdx,
        },
      };
      return newHist;
    });
    setCurrentBranchId(selectedBranch.branchId);
  };

  const goToNext = async (messageId: string) => {
    const conv = history[currentIndex];
    const branches = conv.branchesByEditId?.[messageId] || [];
    const currentIdx = conv.currentBranchIndexByEditId?.[messageId] || 0;

    if (branches.length <= 1) return;
    const nextIdx = (currentIdx + 1) % branches.length;
    const selectedBranch = branches[nextIdx];
    await activateBranch(selectedBranch.branchId);
    setHistory((prev) => {
      const newHist = [...prev];
      newHist[currentIndex] = {
        ...conv,
        messages: selectedBranch.messages,
        currentBranchIndexByEditId: {
          ...(conv.currentBranchIndexByEditId || {}),
          [messageId]: nextIdx,
        },
      };
      return newHist;
    });
    setCurrentBranchId(selectedBranch.branchId);
  };

  // ---------- UI handlers -----------------
  const handleExampleClick = (example: string) => {
    setInputValue(example);
  };

  const toggleAccordion = (value: string) => {
    setOpenAccordions((prev) =>
      prev.includes(value) ? prev.filter((v) => v !== value) : [...prev, value]
    );
  };

  // copy
  const handleCopy = async (id: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setJustCopiedId(id);
      // clear after 1.5s, revert icon
      setTimeout(() => setJustCopiedId(null), 1500);
    } catch (err) {
      console.error("Copy failed", err);
    }
  };

  // upload documents
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    console.log("File selected:", {
      name: file.name,
      size: file.size,
      type: file.type,
      preset: preset,
    });

    setUploading(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("preset", preset);

    // Debug: Log FormData contents
    console.log("FormData contents:");
    for (const [key, value] of formData.entries()) {
      console.log(key, value);
    }

    const {
      data: { session },
      error,
    } = await supabase.auth.getSession();

    const uploadHeaders: Record<string, string> = {};
    if (session?.access_token) {
      uploadHeaders.Authorization = `Bearer ${session.access_token}`;
      console.log("Auth token present");
    } else {
      console.log("No auth token");
    }

    try {
      console.log("Sending request to:", `${API_URL}/api/documents/append`);
      const res = await fetch(`${API_URL}/api/documents/append`, {
        method: "POST",
        headers: uploadHeaders,
        body: formData,
      });

      console.log("Response status:", res.status);
      console.log(
        "Response headers:",
        Object.fromEntries(res.headers.entries())
      );

      if (res.ok) {
        const responseData = await res.json();
        console.log("Success response:", responseData);
        alert("Document ingested successfully");
      } else {
        const errorText = await res.text();
        console.error("Error response:", errorText);

        // Try to parse as JSON for better error info
        try {
          const errorJson = JSON.parse(errorText);
          console.error("Parsed error:", errorJson);
          alert(
            `Failed to ingest document: ${
              errorJson.detail || errorJson.message || errorText
            }`
          );
        } catch {
          alert(`Failed to ingest document: ${errorText}`);
        }
      }
    } catch (error: any) {
      console.error("Network error:", error);
      alert(`Network error: ${error.message}`);
    } finally {
      setUploading(false);
    }
  };

  // chat rename
  const startInlineRename = (id: string, currentTitle: string) => {
    setEditingId(id);
    setEditingText(currentTitle);
    setOpenMenuId(null);
  };

  async function commitInlineRename() {
    const {
      data: { session },
    } = await supabase.auth.getSession();
    if (!editingId) return;

    const trimmed = editingText.trim();
    if (!trimmed) return setEditingId(null);

    if (session?.access_token) {
      headers.Authorization = `Bearer ${session.access_token}`;
    }
    const res = await fetch(`${API_URL}/api/conversations/${editingId}`, {
      method: "PATCH",
      headers,
      body: JSON.stringify({ title: trimmed }),
    });
    if (!res.ok) throw new Error("Could not rename conversation");

    setConversations((prev) =>
      prev.map((c) => (c.id === editingId ? { ...c, title: trimmed } : c))
    );
    setEditingId(null);
    setEditingText("");
  }

  // create new chat
  async function handleNewChat() {
    const {
      data: { session },
    } = await supabase.auth.getSession();

    if (session?.access_token) {
      headers.Authorization = `Bearer ${session.access_token}`;
    }
    const res = await fetch(`${API_URL}/api/conversations`, {
      method: "POST",
      headers,
      body: JSON.stringify({ title: "New Chat" }),
    });
    if (!res.ok) throw new Error("Could not create conversation");
    const newConvo = await res.json();

    // push into conversations list
    setConversations((prev) => [newConvo, ...prev]);

    // initialize an empty message history slot for it
    setHistory((prev) => [
      {
        messages: [],
        originalMessages: [],
        editAtId: undefined,
        branchesByEditId: {},
        currentBranchIndexByEditId: {},
      },
      ...prev,
    ]);
    setCurrentIndex(0);

    return newConvo;
  }

  // delete chat
  const deleteChat = async (deletingId: string | null) => {
    if (!deletingId) return;
    const {
      data: { session },
    } = await supabase.auth.getSession();

    if (session?.access_token) {
      headers.Authorization = `Bearer ${session.access_token}`;
    }
    const res = await fetch(`${API_URL}/api/conversations/${deletingId}`, {
      method: "DELETE",
      headers,
    });
    if (!res.ok) throw new Error("Failed to delete conversation");

    // Get current conversation ID being deleted
    const currentConvId = conversations[currentIndex]?.id;
    const isDeletingCurrentConv = currentConvId === deletingId;

    // remove from both lists by ID
    const newConversations = conversations.filter((c) => c.id !== deletingId);
    setConversations(newConversations);
    setHistory((prev) =>
      prev.filter((_, idx) => conversations[idx]?.id !== deletingId)
    );

    // If deleting current conversation, create new chat or go to first conversation
    if (isDeletingCurrentConv) {
      if (newConversations.length === 0) {
        // No conversations left, create a new one
        const newConvo = await handleNewChat();
        if (newConvo) {
          setConversations([newConvo]);
          setHistory([
            {
              messages: [],
              originalMessages: [],
              editAtId: undefined,
              branchesByEditId: {},
              currentBranchIndexByEditId: {},
            },
          ]);
          setCurrentIndex(0);
        }
      } else {
        // Go to first conversation and load its history
        setCurrentIndex(0);
        try {
          await handleConversationClick(0, newConversations[0].id);
        } catch (err) {
          console.error("Error loading conversation after deletion:", err);
        }
      }
    } else {
      // Adjust current index if needed
      const newIndex = newConversations.findIndex(
        (c) => c.id === currentConvId
      );
      setCurrentIndex(Math.max(0, newIndex));
    }

    setDeletingId(null);
  };

  // --------- search ------------
  const [showSearchModal, setShowSearchModal] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  // search chat handlers
  const filteredResults = conversations.filter((conv) =>
    conv.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleSearchResultClick = (idx: number) => {
    setCurrentIndex(idx);
    setShowSearchModal(false);
    setSearchQuery("");
  };

  // ---------- user menu / logout ------------
  const handleUserMenuToggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    setShowUserMenu((prev) => !prev);
  };

  const handleLogout = async () => {
    await supabase.auth.signOut();
    router.push("/auth/login");
  };

  // + drop down
  const togglePlusDropdown = () => {
    setShowPlusDropdown(!showPlusDropdown);
    setOpenSubmenu(null);
  };

  const toggleSubmenu = (submenuName: string) => {
    setOpenSubmenu(openSubmenu === submenuName ? null : submenuName);
  };

  const selectFeature = (feature: string) => {
    // Remove any existing features from the same category
    const category = feature.split(" - ")[0];
    const filteredFeatures = activeFeatures.filter(
      (f) => !f.startsWith(category)
    );

    // Add the new feature
    setActiveFeatures([...filteredFeatures, feature]);
    setOpenSubmenu(null);
  };

  const clearFeature = (featureToRemove: string) => {
    setActiveFeatures(activeFeatures.filter((f) => f !== featureToRemove));
  };

  const clearAllFeatures = () => {
    setActiveFeatures([]);
  };

  // loading spinner
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div
      className="flex h-screen bg-background text-foreground overflow-hidden"
      onClick={() => {
        setOpenMenuId(null);
        setShowUserMenu(false);
        setShowPlusDropdown(false);
        setOpenSubmenu(null);
      }}
    >
      {/* Left Sidebar */}
      {showLeftSidebar && (
        <div className="w-56 sm:w-64 bg-sidebar border-r border-sidebar-border flex flex-col">
          <div className="p-3 sm:p-4 border-b border-sidebar-border">
            <div className="flex items-center justify-between mb-3 sm:mb-4">
              <div className="flex items-center gap-2">
                <div className="w-16 sm:w-20 h-8 sm:h-10 flex items-center justify-center">
                  <Image
                    src="/Innovision-White-Text.svg"
                    alt="Logo"
                    width={100}
                    height={100}
                    className="object-cover"
                  />
                </div>
              </div>
              <button
                onClick={() => !isAwaitingResponse && setShowLeftSidebar(false)}
                disabled={isAwaitingResponse}
                className={`p-1 rounded transition-colors ${
                  isAwaitingResponse
                    ? "text-sidebar-foreground/50 cursor-not-allowed"
                    : "text-sidebar-foreground hover:text-sidebar-foreground hover:bg-sidebar-accent cursor-pointer"
                }`}
                title={
                  isAwaitingResponse
                    ? "Please wait for current operation to complete"
                    : "Hide left sidebar"
                }
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <button
              onClick={handleNewChat}
              disabled={isAwaitingResponse}
              className={`w-full flex items-center justify-start p-2 rounded transition-colors text-sm sm:text-base ${
                isAwaitingResponse
                  ? "text-sidebar-foreground/50 cursor-not-allowed"
                  : "text-sidebar-foreground hover:text-sidebar-foreground hover:bg-sidebar-accent cursor-pointer"
              }`}
              title={
                isAwaitingResponse
                  ? "Please wait for current operation to complete"
                  : "Start a new chat"
              }
            >
              <Plus className="w-4 h-4 mr-2" />
              New chat
            </button>
            <button
              onClick={() => setShowSearchModal(true)}
              disabled={isAwaitingResponse}
              className={`w-full flex items-center justify-start p-2 rounded mt-2 transition-colors text-sm sm:text-base ${
                isAwaitingResponse
                  ? "text-sidebar-foreground/50 cursor-not-allowed"
                  : "text-sidebar-foreground hover:text-sidebar-foreground hover:bg-sidebar-accent cursor-pointer"
              }`}
              title={
                isAwaitingResponse
                  ? "Please wait for current operation to complete"
                  : "Search through your chats"
              }
            >
              <Search className="w-4 h-4 mr-2" />
              Search chats
            </button>
          </div>

          <div className="flex-1 p-3 sm:p-4 overflow-y-auto sidebar-scrollbar">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs sm:text-sm text-sidebar-foreground">
                Recent Conversations
              </span>
            </div>

            {conversations.length === 0 ? (
              <div className="text-xs sm:text-sm text-sidebar-foreground italic">
                No chats yet
              </div>
            ) : (
              conversations.map((conv, idx) => (
                <div
                  key={conv.id}
                  onClick={() =>
                    !isAwaitingResponse && handleConversationClick(idx, conv.id)
                  }
                  className={`flex items-center justify-between p-2 rounded transition-colors
                    ${
                      idx === currentIndex
                        ? "bg-gray-700 text-white"
                        : isAwaitingResponse
                        ? "text-sidebar-foreground/50 cursor-not-allowed"
                        : "text-sidebar-foreground hover:text-sidebar-foreground hover:bg-sidebar-accent cursor-pointer"
                    }
                  `}
                >
                  {/* Title / Inline Input */}
                  {editingId === conv.id ? (
                    <input
                      type="text"
                      value={editingText}
                      autoFocus
                      onChange={(e) => setEditingText(e.target.value)}
                      onBlur={commitInlineRename}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          commitInlineRename();
                        }
                      }}
                      className="w-full bg-input text-foreground px-2 py-1 rounded border border-border focus:outline-none focus:ring-2 focus:ring-primary"
                    />
                  ) : (
                    <span
                      className={`text-xs sm:text-sm truncate ${
                        isAwaitingResponse
                          ? "text-sidebar-foreground/50 cursor-not-allowed"
                          : "text-sidebar-foreground cursor-pointer"
                      }`}
                      onClick={() =>
                        !isAwaitingResponse &&
                        handleConversationClick(idx, conv.id)
                      }
                      title={
                        isAwaitingResponse
                          ? "Please wait for current operation to complete"
                          : conv.title
                      }
                    >
                      {conv.title}
                    </span>
                  )}

                  {/* 3-dot menu */}
                  <div className="relative ml-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setOpenMenuId(conv.id);
                      }}
                      disabled={isAwaitingResponse}
                      className={`p-1 rounded ${
                        isAwaitingResponse
                          ? "text-sidebar-foreground/50 cursor-not-allowed"
                          : "text-sidebar-foreground hover:text-sidebar-foreground hover:bg-sidebar-accent cursor-pointer"
                      }`}
                      title={
                        isAwaitingResponse
                          ? "Please wait for current operation to complete"
                          : "More options"
                      }
                    >
                      <MoreVertical className="w-4 h-4" />
                    </button>

                    {openMenuId === conv.id && (
                      <div className="absolute right-0 mt-1 w-32 bg-gray-900 border border-sidebar-border rounded shadow-lg z-20">
                        <button
                          className={`w-full text-left px-4 py-2 text-sm flex items-center gap-2 ${
                            isAwaitingResponse
                              ? "text-sidebar-foreground/50 cursor-not-allowed"
                              : "text-sidebar-foreground hover:bg-sidebar-primary cursor-pointer"
                          }`}
                          onClick={() =>
                            !isAwaitingResponse &&
                            startInlineRename(conv.id, conv.title)
                          }
                          disabled={isAwaitingResponse}
                          title={
                            isAwaitingResponse
                              ? "Please wait for current operation to complete"
                              : "Rename conversation"
                          }
                        >
                          <Edit className="w-4 h-4" />
                          Rename
                        </button>
                        <button
                          className={`w-full text-left px-4 py-2 text-sm flex items-center gap-2 ${
                            isAwaitingResponse
                              ? "text-sidebar-foreground/50 cursor-not-allowed"
                              : "text-sidebar-foreground hover:bg-sidebar-primary cursor-pointer"
                          }`}
                          onClick={() => {
                            if (!isAwaitingResponse) {
                              setOpenMenuId(null);
                              setDeletingId(conv.id);
                            }
                          }}
                          disabled={isAwaitingResponse}
                          title={
                            isAwaitingResponse
                              ? "Please wait for current operation to complete"
                              : "Delete conversation"
                          }
                        >
                          Delete
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>

          <div className="p-3 sm:p-4 border-t border-sidebar-border relative">
            {user ? (
              <div>
                <button
                  onClick={handleUserMenuToggle}
                  disabled={isAwaitingResponse}
                  className={`flex items-center justify-between gap-2 w-full focus:outline-none ${
                    isAwaitingResponse ? "cursor-not-allowed opacity-50" : ""
                  }`}
                  title={
                    isAwaitingResponse
                      ? "Please wait for current operation to complete"
                      : "User menu"
                  }
                >
                  <div className="w-12 h-8 bg-black rounded-full flex items-center justify-center">
                    <UserIcon className="w-4 h-4 text-sidebar-foreground" />
                  </div>
                  <span className="block w-full text-left text-xs sm:text-sm text-sidebar-foreground justify-left truncate">
                    {user.user_metadata.full_name}
                  </span>
                  <ChevronDown
                    className={`w-4 h-4 transition-transform text-sidebar-foreground ${
                      showUserMenu ? "rotate-180" : ""
                    }`}
                  />
                </button>

                {showUserMenu && (
                  <div className="absolute bottom-full inset-x-0 ml-2 mb-2 w-9/10 bg-black border border-sidebar-border rounded shadow-lg z-50">
                    <button
                      onClick={openProfileModal}
                      disabled={isAwaitingResponse}
                      className={`w-full text-left px-4 py-2 text-sm ${
                        isAwaitingResponse
                          ? "text-sidebar-foreground/50 cursor-not-allowed"
                          : "text-sidebar-foreground hover:bg-sidebar-primary cursor-pointer"
                      }`}
                      title={
                        isAwaitingResponse
                          ? "Please wait for current operation to complete"
                          : "Manage your profile"
                      }
                    >
                      Manage profile
                    </button>
                    <button
                      onClick={handleLogout}
                      disabled={isAwaitingResponse}
                      className={`w-full text-left px-4 py-2 text-sm ${
                        isAwaitingResponse
                          ? "text-destructive/50 cursor-not-allowed"
                          : "text-destructive hover:bg-sidebar-primary cursor-pointer"
                      }`}
                      title={
                        isAwaitingResponse
                          ? "Please wait for current operation to complete"
                          : "Sign out of your account"
                      }
                    >
                      Log out
                    </button>
                  </div>
                )}

                <ManageProfile
                  isOpen={showProfileModal}
                  onClose={closeProfileModal}
                  user={user}
                />
              </div>
            ) : (
              <div className="flex space-x-2">
                <Link href="/auth/login">
                  <Button
                    variant="secondary"
                    size="lg"
                    className="cursor-pointer text-blue-400 hover:text-blue-300 border-blue-400 hover:border-blue-300"
                  >
                    Log in
                  </Button>
                </Link>
                <Link href="/auth/signup">
                  <Button
                    size="lg"
                    variant="default"
                    className="cursor-pointer bg-blue-600 hover:bg-blue-700 text-white"
                  >
                    Sign up
                  </Button>
                </Link>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
        {/* Header */}
        <div className="h-16 border-b border-border flex items-center justify-between px-4 sm:px-6 flex-shrink-0">
          <div className="flex items-center gap-2 sm:gap-3 flex-wrap">
            {!showLeftSidebar && (
              <button
                onClick={() => !isAwaitingResponse && setShowLeftSidebar(true)}
                disabled={isAwaitingResponse}
                className={`p-2 rounded ${
                  isAwaitingResponse
                    ? "text-gray-500 cursor-not-allowed"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent cursor-pointer"
                }`}
                title={
                  isAwaitingResponse
                    ? "Please wait for current operation to complete"
                    : "Show left sidebar"
                }
              >
                <Menu className="w-4 h-4" />
              </button>
            )}

            {/* Product Name and Model Selection */}
            <div className="flex items-center gap-2">
              <h1 className="text-base sm:text-lg font-semibold text-foreground">
                Comfit Copilot
              </h1>

              {/* Model Selection*/}
              <Select
                value={selectedModel}
                onValueChange={(value) => {
                  if (!isAwaitingResponse) {
                    console.log("DEBUG: Model selection changed to:", value);
                    setSelectedModel(value);
                  }
                }}
                disabled={isAwaitingResponse}
              >
                <SelectTrigger
                  className={`w-32 sm:w-40 bg-transparent border-none shadow-none focus:ring-0 focus:ring-offset-0 ${
                    isAwaitingResponse ? "opacity-50 cursor-not-allowed" : ""
                  }`}
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-black">
                  {models.length === 0 ? (
                    <SelectItem value="none" disabled>
                      (no models available)
                    </SelectItem>
                  ) : (
                    models.map((m) => (
                      <SelectItem key={m} value={m}>
                        {m}
                      </SelectItem>
                    ))
                  )}
                </SelectContent>
              </Select>
            </div>

            {/* Deep Research Agent Button */}
            <div className="relative" data-deep-research-container>
              <button
                onClick={() =>
                  setIsDeepResearchDropdownOpen(!isDeepResearchDropdownOpen)
                }
                disabled={isAwaitingResponse}
                className={`w-36 sm:w-44 bg-input px-3 py-2 rounded-md text-sm font-normal transition-colors text-left flex items-center justify-between whitespace-nowrap ${
                  isAwaitingResponse ? "opacity-50 cursor-not-allowed" : ""
                }`}
                title="Deep Research Agent - Advanced RAG Strategy Options"
              >
                <span>Deep Research Agent</span>
                <svg
                  width="12"
                  height="12"
                  viewBox="0 0 12 12"
                  fill="none"
                  className={`ml-2 transition-transform ${
                    isDeepResearchDropdownOpen ? "rotate-180" : ""
                  }`}
                >
                  <path
                    d="M3 4.5L6 7.5L9 4.5"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>

              {/* Deep Research Agent Dropdown */}
              {isDeepResearchDropdownOpen && (
                <div className="absolute top-full left-0 mt-1 w-48 bg-black border border-gray-600 rounded-md shadow-lg z-50">
                  <div className="py-1">
                    <div className="px-3 py-2 text-xs font-semibold text-gray-400 uppercase tracking-wide">
                      Available RAG Strategies
                    </div>
                    <button
                      onClick={() => {
                        setSelectedRagMethod("RAC Enhanced Hybrid RAG");
                        setIsDeepResearchDropdownOpen(false);
                      }}
                      className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-800 ${
                        selectedRagMethod === "RAC Enhanced Hybrid RAG"
                          ? "text-blue-400 bg-gray-800"
                          : "text-white"
                      }`}
                    >
                      RAC Enhanced Hybrid RAG
                    </button>
                    <button
                      onClick={() => {
                        setSelectedRagMethod("Planning Workflow");
                        setIsDeepResearchDropdownOpen(false);
                      }}
                      className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-800 ${
                        selectedRagMethod === "Planning Workflow"
                          ? "text-blue-400 bg-gray-800"
                          : "text-white"
                      }`}
                    >
                      Planning Workflow
                    </button>
                    <button
                      onClick={() => {
                        setSelectedRagMethod("Multi-Step Query Engine");
                        setIsDeepResearchDropdownOpen(false);
                      }}
                      className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-800 ${
                        selectedRagMethod === "Multi-Step Query Engine"
                          ? "text-blue-400 bg-gray-800"
                          : "text-white"
                      }`}
                    >
                      Multi-Step Query Engine
                    </button>
                    <button
                      onClick={() => {
                        setSelectedRagMethod("Multi-Strategy Workflow");
                        setIsDeepResearchDropdownOpen(false);
                      }}
                      className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-800 ${
                        selectedRagMethod === "Multi-Strategy Workflow"
                          ? "text-blue-400 bg-gray-800"
                          : "text-white"
                      }`}
                    >
                      Multi-Strategy Workflow
                    </button>
                    <button
                      onClick={() => {
                        setSelectedRagMethod("No Specific RAG Method");
                        setIsDeepResearchDropdownOpen(false);
                      }}
                      className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-800 ${
                        selectedRagMethod === "No Specific RAG Method"
                          ? "text-blue-400 bg-gray-800"
                          : "text-white"
                      }`}
                    >
                      No Specific RAG Method
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Retrieval Methods Button */}
            <div className="relative flex items-center gap-2" data-retrieval-container>
              <button
                onClick={() =>
                  setIsRetrievalDropdownOpen(!isRetrievalDropdownOpen)
                }
                disabled={isAwaitingResponse}
                className={`w-32 sm:w-40 bg-input border-none px-3 py-2 rounded-md text-sm font-normal transition-colors text-left flex items-center justify-between whitespace-nowrap ${
                  isAwaitingResponse ? "opacity-50 cursor-not-allowed" : ""
                }`}
                title={`Retrieval Methods - Select retrieval strategy${selectedRetrievalMethod === "local context only" && selectedVectorStore ? `\nCurrent Vector Store: ${selectedVectorStore}` : ""}`}
              >
                <span>Retrieval Methods</span>
                <svg
                  width="12"
                  height="12"
                  viewBox="0 0 12 12"
                  fill="none"
                  className={`ml-2 transition-transform ${
                    isRetrievalDropdownOpen ? "rotate-180" : ""
                  }`}
                >
                  <path
                    d="M3 4.5L6 7.5L9 4.5"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
              
              {/* Vector Store Badge - Show when local retrieval is active */}
              {selectedRetrievalMethod === "local context only" && selectedVectorStore && (
                <div className="hidden lg:flex items-center gap-1 px-2 py-1 bg-blue-600/20 border border-blue-600/50 rounded text-xs text-blue-400">
                  <Database className="w-3 h-3" />
                  <span className="max-w-[100px] truncate" title={selectedVectorStore}>
                    {selectedVectorStore}
                  </span>
                </div>
              )}

              {/* Retrieval Methods Dropdown */}
              {isRetrievalDropdownOpen && (
                <div className="absolute top-full left-0 mt-1 w-80 bg-black border border-gray-600 rounded-md shadow-lg z-50 max-h-96 overflow-y-auto">
                  <div className="py-1">
                    <div className="px-3 py-2 text-xs font-semibold text-gray-400 uppercase tracking-wide border-b border-gray-700">
                      Retrieval Methods
                    </div>
                    
                    {/* Local Context Only Button */}
                    <button
                      onClick={() => {
                        setSelectedRetrievalMethod("local context only");
                        setIsVectorStoreDropdownOpen(!isVectorStoreDropdownOpen);
                      }}
                      className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-800 flex items-center justify-between ${
                        selectedRetrievalMethod === "local context only"
                          ? "text-blue-400 bg-gray-800"
                          : "text-white"
                      }`}
                    >
                      <span>Local Context Only</span>
                      <svg
                        width="12"
                        height="12"
                        viewBox="0 0 12 12"
                        fill="none"
                        className={`transition-transform ${
                          isVectorStoreDropdownOpen ? "rotate-180" : ""
                        }`}
                      >
                        <path
                          d="M3 4.5L6 7.5L9 4.5"
                          stroke="currentColor"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                    </button>
                    
                    {/* Vector Store Selection - Shown inline when expanded */}
                    {isVectorStoreDropdownOpen && (
                      <div className="bg-gray-900 border-t border-gray-700">
                        <div className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wide">
                          Select Vector Store
                        </div>
                        {Object.entries(vectorStores.categories || {}).map(([category, files]) => (
                          <div key={category} className="mb-2">
                            <div className="px-3 py-1 text-xs font-semibold text-gray-600 uppercase tracking-wider">
                              {category}
                            </div>
                            {(files as string[]).map((fileName) => (
                              <button
                                key={fileName}
                                onClick={() => {
                                  setSelectedVectorStore(fileName);
                                  setIsVectorStoreDropdownOpen(false);
                                  setIsRetrievalDropdownOpen(false);
                                }}
                                disabled={isAwaitingResponse}
                                className={`w-full flex items-center justify-between px-4 py-2 text-sm transition-colors ${
                                  isAwaitingResponse
                                    ? "text-gray-500 cursor-not-allowed opacity-50"
                                    : selectedVectorStore === fileName
                                    ? "bg-blue-600 text-white"
                                    : "text-white hover:bg-gray-800 cursor-pointer"
                                }`}
                                title={
                                  isAwaitingResponse
                                    ? "Please wait for current operation to complete"
                                    : `Select ${fileName}`
                                }
                              >
                                <span className="text-left truncate text-xs">{fileName}</span>
                                {selectedVectorStore === fileName && (
                                  <Check className="w-3 h-3 flex-shrink-0" />
                                )}
                              </button>
                            ))}
                          </div>
                        ))}
                        {Object.keys(vectorStores.categories || {}).length === 0 && (
                          <div className="px-3 py-2 text-sm text-gray-500 italic">
                            No vector stores available
                          </div>
                        )}
                      </div>
                    )}
                    
                    {/* Divider */}
                    <div className="border-t border-gray-700 my-1"></div>
                    
                    <button
                      onClick={() => {
                        setSelectedRetrievalMethod("Web searched context only");
                        setIsVectorStoreDropdownOpen(false);
                        setIsRetrievalDropdownOpen(false);
                      }}
                      className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-800 ${
                        selectedRetrievalMethod === "Web searched context only"
                          ? "text-blue-400 bg-gray-800"
                          : "text-white"
                      }`}
                    >
                      Web Searched Context Only
                    </button>
                    <button
                      onClick={() => {
                        setSelectedRetrievalMethod("Hybrid context");
                        setIsVectorStoreDropdownOpen(false);
                        setIsRetrievalDropdownOpen(false);
                      }}
                      className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-800 ${
                        selectedRetrievalMethod === "Hybrid context"
                          ? "text-blue-400 bg-gray-800"
                          : "text-white"
                      }`}
                    >
                      Hybrid Context
                    </button>
                    <button
                      onClick={() => {
                        setSelectedRetrievalMethod("Smart retrieval");
                        setIsVectorStoreDropdownOpen(false);
                        setIsRetrievalDropdownOpen(false);
                      }}
                      className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-800 ${
                        selectedRetrievalMethod === "Smart retrieval"
                          ? "text-blue-400 bg-gray-800"
                          : "text-white"
                      }`}
                    >
                      Smart Retrieval
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {!showRightSidebar && (
            <button
              onClick={() => !isAwaitingResponse && setShowRightSidebar(true)}
              disabled={isAwaitingResponse}
              className={`p-2 rounded flex items-center gap-2 ${
                isAwaitingResponse
                  ? "text-gray-500 cursor-not-allowed"
                  : "text-muted-foreground hover:text-foreground hover:bg-accent cursor-pointer"
              }`}
              title={
                isAwaitingResponse
                  ? "Please wait for current operation to complete"
                  : "Show right sidebar"
              }
            >
              <Settings className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Chat Area */}
        <div className="flex-1 flex flex-col min-h-0">
          {isWelcomeState ? (
            <div className="flex-1 flex flex-col items-center justify-center p-4 sm:p-8">
              <h1 className="text-xl sm:text-2xl lg:text-3xl font-light mb-6 sm:mb-8 lg:mb-12 text-center px-4">
                What would you like to know?
              </h1>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4 mb-6 sm:mb-8 max-w-6xl mx-auto px-4 w-full">
                <button
                  className={`border border-border px-3 sm:px-4 py-2 rounded text-left text-sm sm:text-base ${
                    isAwaitingResponse
                      ? "text-gray-500 cursor-not-allowed"
                      : "text-muted-foreground hover:bg-accent cursor-pointer"
                  }`}
                  onClick={() =>
                    !isAwaitingResponse &&
                    handleExampleClick(
                      "What is your name and who developed you?"
                    )
                  }
                  disabled={isAwaitingResponse}
                  title={
                    isAwaitingResponse
                      ? "Please wait for current operation to complete"
                      : "Ask about the AI's identity"
                  }
                >
                  What is your name and who developed you?
                </button>
                <button
                  className={`border border-border px-3 sm:px-4 py-2 rounded text-left text-sm sm:text-base ${
                    isAwaitingResponse
                      ? "text-gray-500 cursor-not-allowed"
                      : "text-muted-foreground hover:bg-accent cursor-pointer"
                  }`}
                  onClick={() =>
                    !isAwaitingResponse &&
                    handleExampleClick(
                      "Why is bending your knees before lifting safer than keeping your legs straight?"
                    )
                  }
                  disabled={isAwaitingResponse}
                  title={
                    isAwaitingResponse
                      ? "Please wait for current operation to complete"
                      : "Ask about lifting safety"
                  }
                >
                  Why is bending your knees before lifting safer than keeping
                  your legs straight?
                </button>
                <button
                  className={`border border-border px-3 sm:px-4 py-2 rounded text-left text-sm sm:text-base ${
                    isAwaitingResponse
                      ? "text-gray-500 cursor-not-allowed"
                      : "text-muted-foreground hover:bg-accent cursor-pointer"
                  }`}
                  onClick={() =>
                    !isAwaitingResponse &&
                    handleExampleClick(
                      "Explain why slipping on a wet surface leads to a fall—what forces and frictional changes are at play?"
                    )
                  }
                  disabled={isAwaitingResponse}
                  title={
                    isAwaitingResponse
                      ? "Please wait for current operation to complete"
                      : "Ask about physics of slipping"
                  }
                >
                  Explain why slipping on a wet surface leads to a fall—what
                  forces and frictional changes are at play?
                </button>
              </div>
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto p-4 sm:p-6 chat-scrollbar">
              <div className="max-w-4xl mx-auto space-y-4 sm:space-y-6">
                {(() => {
                  console.log(
                    "DEBUG: Rendering chat area with currentMessages:",
                    {
                      messagesCount: currentMessages.length,
                      messages: currentMessages.map((m) => ({
                        id: m.id,
                        sender: m.sender,
                        contentLength: m.content?.length || 0,
                      })),
                      currentBranchId,
                      historyIndex: currentIndex,
                    }
                  );
                  return null;
                })()}
                {currentMessages.map((message, idx) => {
                  // Debug: Log message details for AI messages
                  if (
                    message.sender === "assistant" ||
                    message.sender === "ai"
                  ) {
                    console.log("DEBUG: Rendering AI message:", {
                      id: message.id,
                      sender: message.sender,
                      content: message.content.substring(0, 50) + "...",
                      thinkingTime: message.thinkingTime,
                      hasThinkingTime: message.thinkingTime != null,
                    });
                  }

                  console.log("DEBUG: Rendering message:", {
                    idx,
                    id: message.id,
                    sender: message.sender,
                    contentLength: message.content?.length || 0,
                    contentPreview: message.content?.substring(0, 50) + "...",
                  });

                  return (
                    <div
                      key={message.id}
                      className={`flex gap-3 sm:gap-4 ${
                        message.sender === "assistant" ||
                        message.sender === "ai"
                          ? "w-full"
                          : ""
                      }`}
                    >
                      {message.sender === "user" ? (
                        <div className="ml-auto max-w-[280px] sm:max-w-xs lg:max-w-md flex flex-col items-end">
                          {/* User bubble */}
                          <div className="bg-gray-600 text-white p-3 rounded-2xl rounded-br-md whitespace-pre-wrap break-words max-w-full shadow-sm">
                            {editingId === message.id ? (
                              <div className="space-y-2 w-full max-w-[280px] sm:max-w-xs lg:max-w-md">
                                <textarea
                                  ref={editTextareaRef}
                                  value={editingText}
                                  onChange={(e) =>
                                    setEditingText(e.target.value)
                                  }
                                  disabled={isAwaitingResponse}
                                  className={`w-full p-2 rounded border border-border focus:outline-none focus:ring-2 focus:ring-primary resize-none overflow-hidden ${
                                    isAwaitingResponse
                                      ? "bg-gray-500 text-gray-300 cursor-not-allowed"
                                      : "bg-input text-foreground"
                                  }`}
                                  rows={1}
                                  autoFocus
                                />
                                <div className="flex gap-2">
                                  <button
                                    type="button"
                                    onClick={() => commitEdit(message.id)}
                                    disabled={isAwaitingResponse}
                                    className={`px-3 py-1 rounded text-sm flex items-center gap-1 transition-all duration-200 active:scale-95 shadow-sm hover:shadow-md ${
                                      isAwaitingResponse
                                        ? "bg-gray-500 text-gray-300 cursor-not-allowed"
                                        : "bg-green-600 hover:bg-green-700 active:bg-green-800 text-white cursor-pointer"
                                    }`}
                                  >
                                    <Check className="w-3 h-3" />
                                    {isAwaitingResponse ? "Saving..." : "Save"}
                                  </button>
                                  <button
                                    type="button"
                                    onClick={cancelEdit}
                                    disabled={isAwaitingResponse}
                                    className={`px-3 py-1 rounded text-sm transition-colors ${
                                      isAwaitingResponse
                                        ? "bg-gray-500 text-gray-300 cursor-not-allowed"
                                        : "bg-muted hover:bg-muted-foreground active:bg-muted/80 text-foreground cursor-pointer"
                                    }`}
                                  >
                                    Cancel
                                  </button>
                                </div>
                              </div>
                            ) : (
                              message.content
                            )}
                          </div>

                          {/* Action icons + (arrows if this was the edited user bubble) */}
                          {editingId !== message.id && (
                            <div className="space-y-1 mt-1">
                              {/*copy + edit */}
                              <div className="flex items-center gap-2.5 pt-1 text-xs text-muted-foreground mt-1">
                                <button
                                  onClick={() =>
                                    handleCopy(message.id, message.content)
                                  }
                                  disabled={isAwaitingResponse}
                                  className={`hover:text-foreground cursor-pointer ${
                                    isAwaitingResponse
                                      ? "opacity-50 cursor-not-allowed"
                                      : ""
                                  }`}
                                  title={
                                    isAwaitingResponse
                                      ? "Please wait for current operation to complete"
                                      : "Copy message"
                                  }
                                >
                                  {justCopiedId === message.id ? (
                                    <Check className="w-4 h-4" />
                                  ) : (
                                    <Copy className="w-4 h-4" />
                                  )}
                                </button>
                                <button
                                  className="hover:text-foreground cursor-pointer"
                                  onClick={() =>
                                    startEdit(message.id, message.content)
                                  }
                                  disabled={
                                    (editingId !== null &&
                                      editingId !== message.id) ||
                                    isAwaitingResponse
                                  }
                                  title={
                                    editingId !== null &&
                                    editingId !== message.id
                                      ? "Finish editing current message first"
                                      : isAwaitingResponse
                                      ? "Please wait for current operation to complete"
                                      : "Edit message"
                                  }
                                >
                                  <Edit
                                    className={`w-4 h-4 ${
                                      (editingId !== null &&
                                        editingId !== message.id) ||
                                      isAwaitingResponse
                                        ? "text-muted-foreground opacity-50"
                                        : ""
                                    }`}
                                  />
                                </button>

                                {/*arrows: show arrows if the user message has multiple branches */}
                                {hasMultipleBranches(message.id) && (
                                  <div className="flex items-center gap-1">
                                    <button
                                      onClick={() => goToPrev(message.id)}
                                      disabled={
                                        getBranchIndex(message.id) === 0 ||
                                        isAwaitingResponse
                                      }
                                      className={`p-1 rounded cursor-pointer ${
                                        getBranchIndex(message.id) === 0 ||
                                        isAwaitingResponse
                                          ? "text-muted-foreground cursor-not-allowed"
                                          : "text-muted-foreground hover:text-foreground hover:bg-accent"
                                      }`}
                                      title={
                                        isAwaitingResponse
                                          ? "Please wait for current operation to complete"
                                          : "Go to previous branch"
                                      }
                                    >
                                      <ArrowLeft className="w-4 h-4" />
                                    </button>
                                    <span className="text-xs text-muted-foreground px-1">
                                      {getBranchIndex(message.id) + 1} /{" "}
                                      {history[currentIndex].branchesByEditId?.[
                                        message.id
                                      ]?.length ?? 1}
                                    </span>
                                    <button
                                      onClick={() => goToNext(message.id)}
                                      disabled={
                                        getBranchIndex(message.id) + 1 ===
                                          (history[currentIndex]
                                            .branchesByEditId?.[message.id]
                                            ?.length ?? 0) || isAwaitingResponse
                                      }
                                      className={`p-1 rounded cursor-pointer ${
                                        getBranchIndex(message.id) + 1 ===
                                          (history[currentIndex]
                                            .branchesByEditId?.[message.id]
                                            ?.length ?? 0) || isAwaitingResponse
                                          ? "text-muted-foreground cursor-not-allowed"
                                          : "text-muted-foreground hover:text-foreground hover:bg-accent"
                                      }`}
                                      title={
                                        isAwaitingResponse
                                          ? "Please wait for current operation to complete"
                                          : "Go to next branch"
                                      }
                                    >
                                      <ArrowRight className="w-4 h-4" />
                                    </button>
                                  </div>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (
                        // AI message - full width like modern chatbots
                        (message.sender === "assistant" ||
                          message.sender === "ai") && (
                          <div className="w-full max-w-none px-4">
                            {message.isThinking ? (
                              // Show only thinking indicator during regeneration
                              <div className="text-white py-3 w-full px-0">
                                <div className="flex items-center gap-2">
                                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-pulse"></div>
                                  <div
                                    className="w-2 h-2 bg-gray-500 rounded-full animate-pulse"
                                    style={{ animationDelay: "0.2s" }}
                                  ></div>
                                  <div
                                    className="w-2 h-2 bg-gray-500 rounded-full animate-pulse"
                                    style={{ animationDelay: "0.4s" }}
                                  ></div>
                                  <div className="flex flex-col ml-2 flex-1">
                                    <div className="flex items-center justify-between">
                                      <span className="text-gray-400 text-sm">
                                        {currentThinkingStep || "Thinking..."}
                                      </span>
                                      {thinkingLogs.length > 0 && (
                                        <button
                                          onClick={() => setShowDetailedThinking(!showDetailedThinking)}
                                          className="text-xs text-blue-400 hover:text-blue-300 ml-2"
                                        >
                                          {showDetailedThinking ? "Hide Details ▲" : "Show Details ▼"}
                                        </button>
                                      )}
                                    </div>
                                    {thinkingDetails && !showDetailedThinking && (
                                      <span className="text-gray-500 text-xs mt-1">
                                        {thinkingDetails}
                                      </span>
                                    )}
                                    {showDetailedThinking && thinkingLogs.length > 0 && (
                                      <div className="mt-2 max-h-96 overflow-y-auto bg-gray-800 rounded p-2 text-xs font-mono">
                                        {thinkingLogs.map((log, idx) => (
                                          <div key={idx} className="text-gray-300 py-0.5 border-b border-gray-700 last:border-0">
                                            {log}
                                          </div>
                                        ))}
                                        <div ref={thinkingLogsEndRef} />
                                      </div>
                                    )}
                                  </div>
                                </div>
                              </div>
                            ) : (
                              // Show normal message content and buttons
                              <>
                                {(() => {
                                  const html = message.content;
                                  const thinkMatch = html.match(
                                    /<think>([\s\S]*?)<\/think>/i
                                  );
                                  const thoughtHtml = thinkMatch?.[1] ?? "";
                                  const mainHtml = html
                                    .replace(/<think>[\s\S]*?<\/think>/i, "")
                                    .replace(
                                      /###\s*Final\s*Response\s*\n*/gi,
                                      ""
                                    )
                                    .replace(/^\r?\n/, "");
                                  
                                  // Check if message has images
                                  const hasImages = message.images && message.images.length > 0;
                                  
                                  // Check for new thinkingContent field (from backend)
                                  const hasThinkingContent = message.thinkingContent && message.thinkingContent.trim().length > 0;
                                  
                                  return (
                                    <div>
                                      {/* Show thinking process toggle if thinking content exists */}
                                      {(hasThinkingContent || thoughtHtml) && (
                                        <button
                                          onClick={() =>
                                            toggleThoughts(message.id)
                                          }
                                          className="flex items-center gap-2 text-blue-400 hover:text-blue-300 mt-2 mr-1 mb-2 text-sm hover:underline cursor-pointer transition-colors"
                                        >
                                          <Brain className="w-4 h-4" />
                                          {showThoughts[message.id]
                                            ? "Hide thinking process"
                                            : "Show thinking process"}
                                        </button>
                                      )}
                                      {/* Show thinking section if toggled */}
                                      {hasThinkingContent && showThoughts[message.id] && (
                                        <div className="bg-gray-800/50 border border-gray-700 rounded-lg py-3 px-4 mb-4">
                                          <div className="flex items-center gap-2 mb-2">
                                            <Brain className="w-4 h-4 text-blue-400" />
                                            <span className="text-sm font-semibold text-blue-400">Thinking Process</span>
                                          </div>
                                          <div className="text-gray-300 text-sm leading-relaxed whitespace-pre-wrap">
                                            {message.thinkingContent}
                                          </div>
                                        </div>
                                      )}
                                      {/* Show legacy reasoning section if toggled (backward compatibility) */}
                                      {thoughtHtml &&
                                        !hasThinkingContent &&
                                        showThoughts[message.id] && (
                                          <div className="text-gray-300 py-2 px-0 whitespace-pre-wrap border-l-2 border-gray-600 pl-4 mb-3">
                                            <div className="text-sm opacity-90">
                                              <div
                                                dangerouslySetInnerHTML={{
                                                  __html: thoughtHtml,
                                                }}
                                              />
                                            </div>
                                          </div>
                                        )}
                                      {/* Always show text response */}
                                      <FormattedContent
                                        html={mainHtml}
                                        className="text-white py-2 px-0 w-full custom-list max-w-full leading-relaxed"
                                      />

                                      {/* Render retrieved images below the text */}
                                      {hasImages && (
                                          <div className="flex flex-wrap gap-4 mt-4 mb-4">
                                            {message.images.map((img, i) => (
                                              <div key={i} className="max-w-md rounded-lg shadow-lg">
                                                <img
                                                  src={img.url}
                                                  alt={img.title}
                                                  className="w-full h-auto object-contain max-h-96 rounded-lg"
                                                  onError={(e: any) => {
                                                    console.error("Failed to load image:", img.url);
                                                    e.currentTarget.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300"><rect fill="%23333"/><text x="50%" y="50%" text-anchor="middle" fill="white">Image not found</text></svg>';
                                                  }}
                                                />
                                              </div>
                                            ))}
                                          </div>
                                        )}
                                    </div>
                                  );
                                })()}
                                {/* Show thinking time if available */}
                                {message.thinkingTime != null && (
                                  <div className="text-muted-foreground text-sm mt-2">
                                    Thought for{" "}
                                    {(message.thinkingTime / 1000).toFixed(2)}s
                                  </div>
                                )}

                                {/* Buttons */}
                                <div className="flex gap-2.5 pt-3 text-xs text-muted-foreground">
                                  <button
                                    onClick={() =>
                                      handleFeedback(message.id, 0)
                                    }
                                    disabled={isAwaitingResponse}
                                    className={`hover:text-foreground cursor-pointer ${
                                      isAwaitingResponse
                                        ? "opacity-50 cursor-not-allowed"
                                        : ""
                                    }`}
                                    title={
                                      isAwaitingResponse
                                        ? "Please wait for current operation to complete"
                                        : "Thumbs up"
                                    }
                                  >
                                    <ThumbsUp
                                      className={`w-4 h-4 ${
                                        message.feedback === 0
                                          ? "text-green-400"
                                          : "text-muted-foreground hover:text-foreground"
                                      }`}
                                    />
                                  </button>
                                  <button
                                    onClick={() =>
                                      handleFeedback(message.id, 1)
                                    }
                                    disabled={isAwaitingResponse}
                                    className={`hover:text-foreground cursor-pointer ${
                                      isAwaitingResponse
                                        ? "opacity-50 cursor-not-allowed"
                                        : ""
                                    }`}
                                    title={
                                      isAwaitingResponse
                                        ? "Please wait for current operation to complete"
                                        : "Thumbs down"
                                    }
                                  >
                                    <ThumbsDown
                                      className={`w-4 h-4 ${
                                        message.feedback === 1
                                          ? "text-red-400"
                                          : "text-muted-foreground hover:text-foreground"
                                      }`}
                                    />
                                  </button>
                                  <button
                                    onClick={() => handleRegenerate(message.id)}
                                    title={
                                      message.isThinking
                                        ? "Regenerating..."
                                        : isAwaitingResponse
                                        ? "Please wait for current operation to complete"
                                        : "Regenerate response"
                                    }
                                    className={`hover:text-foreground cursor-pointer transition-colors ${
                                      message.isThinking
                                        ? "animate-spin text-blue-500"
                                        : ""
                                    }`}
                                    disabled={
                                      isRegenerating ||
                                      message.isThinking ||
                                      isAwaitingResponse
                                    }
                                  >
                                    <RefreshCw className="w-4 h-4" />
                                  </button>
                                  <button
                                    onClick={() =>
                                      handleCopy(message.id, message.content)
                                    }
                                    disabled={isAwaitingResponse}
                                    className={`hover:text-foreground cursor-pointer ${
                                      isAwaitingResponse
                                        ? "opacity-50 cursor-not-allowed"
                                        : ""
                                    }`}
                                    title={
                                      isAwaitingResponse
                                        ? "Please wait for current operation to complete"
                                        : "Copy message"
                                    }
                                  >
                                    {justCopiedId === message.id ? (
                                      <Check className="w-4 h-4" />
                                    ) : (
                                      <Copy className="w-4 h-4" />
                                    )}
                                  </button>
                                </div>
                              </>
                            )}
                          </div>
                        )
                      )}
                    </div>
                  );
                })}

                {isAwaitingResponse && (
                  <div className="flex gap-3 sm:gap-4">
                    <div className="max-w-[280px] sm:max-w-xs lg:max-w-md">
                      <div className="bg-gray-100 text-gray-800 p-3 rounded-2xl rounded-bl-md w-full shadow-sm">
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 bg-gray-500 rounded-full animate-pulse"></div>
                          <div
                            className="w-2 h-2 bg-gray-500 rounded-full animate-pulse"
                            style={{ animationDelay: "0.2s" }}
                          ></div>
                          <div
                            className="w-2 h-2 bg-gray-500 rounded-full animate-pulse"
                            style={{ animationDelay: "0.4s" }}
                          ></div>
                          <div className="flex flex-col ml-2 flex-1">
                            <div className="flex items-center justify-between">
                              <span className="text-gray-400 text-sm">
                                {currentThinkingStep || "Thinking..."}
                              </span>
                              {thinkingLogs.length > 0 && (
                                <button
                                  onClick={() => setShowDetailedThinking(!showDetailedThinking)}
                                  className="text-xs text-blue-400 hover:text-blue-300 ml-2"
                                >
                                  {showDetailedThinking ? "Hide Details ▲" : "Show Details ▼"}
                                </button>
                              )}
                            </div>
                            {thinkingDetails && !showDetailedThinking && (
                              <span className="text-gray-500 text-xs mt-1">
                                {thinkingDetails}
                              </span>
                            )}
                            {showDetailedThinking && thinkingLogs.length > 0 && (
                              <div className="mt-2 max-h-96 overflow-y-auto bg-gray-800 rounded p-2 text-xs font-mono">
                                {thinkingLogs.map((log, idx) => (
                                  <div key={idx} className="text-gray-300 py-0.5 border-b border-gray-700 last:border-0">
                                    {log}
                                  </div>
                                ))}
                                <div ref={thinkingLogsEndRef} />
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            </div>
          )}

          {/* Input Area*/}
          <div className="p-4 sm:p-6 border-t border-border flex-shrink-0">
            <div className="max-w-4xl mx-auto">
              {/* Active Features Display */}
              <div className="mb-3 flex flex-wrap gap-2">
                {/* Engine Type Indicator */}
                <div className="flex items-center gap-2 bg-blue-600 text-white px-3 py-1 rounded-full text-sm">
                  {engineType === "chat" ? (
                    <>
                      <MessageSquare className="w-3 h-3" />
                      <span>Chat Engine</span>
                    </>
                  ) : (
                    <>
                      <Database className="w-3 h-3" />
                      <span>Query Engine</span>
                    </>
                  )}
                </div>

                {/* Other Active Features */}
                {activeFeatures.length > 0 && (
                  <>
                    {activeFeatures.map((feature) => (
                      <div
                        key={feature}
                        className="flex items-center gap-2 bg-primary text-primary-foreground px-3 py-1 rounded-full text-sm"
                      >
                        <span>{feature}</span>
                        <button
                          onClick={() => clearFeature(feature)}
                          disabled={isAwaitingResponse}
                          className={`rounded-full p-0.5 transition-colors ${
                            isAwaitingResponse
                              ? "opacity-50 cursor-not-allowed"
                              : "hover:bg-primary/80"
                          }`}
                          title={
                            isAwaitingResponse
                              ? "Please wait for current operation to complete"
                              : `Remove ${feature}`
                          }
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                    ))}
                    <button
                      onClick={clearAllFeatures}
                      disabled={isAwaitingResponse}
                      className={`text-sm underline transition-colors ${
                        isAwaitingResponse
                          ? "text-gray-500 cursor-not-allowed"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                      title={
                        isAwaitingResponse
                          ? "Please wait for current operation to complete"
                          : "Clear all features"
                      }
                    >
                      Clear all
                    </button>
                  </>
                )}
              </div>

              <div className="relative min-h-[48px] bg-input border border-border rounded-lg">
                {/* Text content area - constrained to not overlap with icons */}
                <div className="relative" style={{ paddingBottom: "40px" }}>
                  <textarea
                    value={inputValue}
                    onChange={(e) => {
                      setInputValue(e.target.value);
                      const ta = e.target;
                      ta.style.height = "auto";
                      const newHeight = Math.min(ta.scrollHeight, 160);
                      ta.style.height = newHeight + "px";

                      // Show scrollbar only when content exceeds max height
                      if (ta.scrollHeight > 160) {
                        ta.style.overflowY = "scroll";
                      } else {
                        ta.style.overflowY = "hidden";
                      }
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        handleSendMessage();
                      }
                    }}
                    placeholder={
                      isAwaitingResponse
                        ? "Please wait..."
                        : "Ask Comfit Copilot..."
                    }
                    rows={1}
                    disabled={isAwaitingResponse}
                    className={`w-full bg-transparent px-4 pt-3 pb-2 resize-none border-none outline-none ${
                      isAwaitingResponse
                        ? "text-gray-400 cursor-not-allowed"
                        : "text-foreground"
                    }`}
                    style={{
                      lineHeight: "1.5",
                      minHeight: "44px",
                      maxHeight: "160px",
                      overflowY: "hidden",
                    }}
                  />
                </div>

                {/* Icon area - absolutely positioned at bottom */}
                <div className="absolute inset-x-0 bottom-2 flex justify-between items-center px-3 pointer-events-none">
                  <div className="flex items-center gap-2 pointer-events-auto">
                    {/* Plus dropdown */}
                    <div className="relative">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          togglePlusDropdown();
                        }}
                        disabled={isAwaitingResponse}
                        className={`flex items-center justify-center w-8 h-8 rounded transition-colors duration-150 ${
                          isAwaitingResponse
                            ? "cursor-not-allowed opacity-50"
                            : "cursor-pointer hover:bg-accent"
                        }`}
                        title={
                          isAwaitingResponse
                            ? "Please wait for current operation to complete"
                            : "More options"
                        }
                      >
                        <Plus className="w-5 h-5 text-muted-foreground" />
                      </button>

                      {showPlusDropdown && (
                        <div
                          className="absolute bottom-full left-0 mb-2 w-64 bg-black border border-border rounded-lg shadow-lg z-50"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <div className="p-2">
                            {/* Engine Type Toggle */}
                            <div className="mb-2 p-2 bg-gray-900 rounded">
                              <div className="text-xs text-gray-400 mb-1">
                                Engine Type
                              </div>
                              <div className="flex gap-1">
                                <button
                                  onClick={() => setEngineType("chat")}
                                  disabled={isAwaitingResponse}
                                  className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-medium transition-colors ${
                                    engineType === "chat"
                                      ? "bg-blue-600 text-white"
                                      : isAwaitingResponse
                                      ? "text-gray-500 cursor-not-allowed"
                                      : "text-gray-300 hover:text-white hover:bg-gray-700"
                                  }`}
                                >
                                  <MessageSquare className="w-3 h-3" />
                                  Chat
                                </button>
                                <button
                                  onClick={() => setEngineType("query")}
                                  disabled={isAwaitingResponse}
                                  className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-medium transition-colors ${
                                    engineType === "query"
                                      ? "bg-blue-600 text-white"
                                      : isAwaitingResponse
                                      ? "text-gray-500 cursor-not-allowed"
                                      : "text-gray-300 hover:text-white hover:bg-gray-700"
                                  }`}
                                >
                                  <Database className="w-3 h-3" />
                                  Query
                                </button>
                              </div>
                              <div className="text-xs text-gray-500 mt-1">
                                {engineType === "chat"
                                  ? "Maintains conversation context"
                                  : "More accurate but no chat context"}
                              </div>
                            </div>

                            {/* Vector Store */}
                            <div className="relative">
                              <button
                                onClick={() => toggleSubmenu("vector-store")}
                                disabled={isAwaitingResponse}
                                className={`w-full flex items-center gap-2 px-3 py-2 text-sm rounded transition-colors ${
                                  isAwaitingResponse
                                    ? "text-muted-foreground cursor-not-allowed opacity-50"
                                    : "text-foreground hover:bg-accent cursor-pointer"
                                }`}
                                title={
                                  isAwaitingResponse
                                    ? "Please wait for current operation to complete"
                                    : "Vector Store options - Select document source"
                                }
                              >
                                <VectorSquare className="w-4 h-4" />
                                <span>Vector Store: {selectedVectorStore || 'None'}</span>
                                <ChevronRight className="w-4 h-4 ml-auto" />
                              </button>
                              {openSubmenu === "vector-store" && (
                                <div className="absolute left-full top-0 ml-2 w-80 bg-black border border-border rounded-lg shadow-lg max-h-96 overflow-y-auto">
                                  <div className="p-2">
                                    {Object.entries(vectorStores.categories || {}).map(([category, files]) => (
                                      <div key={category} className="mb-2">
                                        <div className="px-3 py-1 text-xs font-semibold text-gray-400 uppercase tracking-wider">
                                          {category}
                                        </div>
                                        {(files as string[]).map((fileName) => (
                                          <button
                                            key={fileName}
                                            onClick={() => {
                                              setSelectedVectorStore(fileName);
                                              setOpenSubmenu(null);
                                            }}
                                            disabled={isAwaitingResponse}
                                            className={`w-full flex items-center justify-between px-3 py-2 text-sm rounded transition-colors ${
                                              isAwaitingResponse
                                                ? "text-muted-foreground cursor-not-allowed opacity-50"
                                                : selectedVectorStore === fileName
                                                ? "bg-accent text-foreground"
                                                : "text-foreground hover:bg-accent/50 cursor-pointer"
                                            }`}
                                            title={
                                              isAwaitingResponse
                                                ? "Please wait for current operation to complete"
                                                : `Select ${fileName}`
                                            }
                                          >
                                            <span className="text-left">{fileName}</span>
                                            {selectedVectorStore === fileName && (
                                              <Check className="w-4 h-4" />
                                            )}
                                          </button>
                                        ))}
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>

                            {/* Add Files */}
                            <label
                              htmlFor="file-upload"
                              className={`w-full flex items-center gap-2 px-3 py-2 text-sm rounded transition-colors ${
                                uploading || isAwaitingResponse
                                  ? "text-muted-foreground cursor-not-allowed opacity-50"
                                  : "text-foreground hover:bg-accent cursor-pointer"
                              }`}
                            >
                              <Paperclip className="w-4 h-4" />
                              <span>Add Files</span>
                              <input
                                id="file-upload"
                                type="file"
                                accept=".pdf,.txt"
                                onChange={handleFileChange}
                                disabled={uploading || isAwaitingResponse}
                                className="sr-only"
                              />
                            </label>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Voice Mode Icon and send/cancel button */}
                  <div className="flex items-center gap-2 pointer-events-auto">
                    <button
                      className={`flex items-center justify-center w-8 h-8 rounded transition-colors duration-150 ${
                        isAwaitingResponse
                          ? "cursor-not-allowed opacity-50"
                          : "cursor-pointer hover:bg-accent"
                      }`}
                      disabled={isAwaitingResponse}
                      title={
                        isAwaitingResponse
                          ? "Please wait for current operation to complete"
                          : "Voice input"
                      }
                    >
                      <Mic className="w-5 h-5 text-muted-foreground" />
                    </button>

                    <button
                      onClick={
                        isAwaitingResponse ? handleCancel : handleSendMessage
                      }
                      disabled={
                        limitReached ||
                        (!isAwaitingResponse && !inputValue.trim()) ||
                        isAwaitingResponse
                      }
                      title={
                        isAwaitingResponse
                          ? "Cancel AI response"
                          : inputValue.trim()
                          ? "Send message"
                          : "Type a message to send"
                      }
                      className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-150 ${
                        isAwaitingResponse
                          ? "bg-red-500 hover:bg-red-600 text-white shadow-lg"
                          : inputValue.trim()
                          ? "bg-blue-600 hover:bg-blue-700 text-white shadow-md hover:shadow-lg"
                          : "bg-gray-400 text-gray-600 cursor-not-allowed"
                      }`}
                    >
                      {isAwaitingResponse ? (
                        <X size={20} className="animate-pulse" />
                      ) : (
                        <ArrowUp size={20} />
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Right Sidebar*/}
      {showRightSidebar && (
        <div className="w-64 sm:w-80 bg-sidebar border-l border-sidebar-border flex flex-col max-h-screen overflow-y-auto sidebar-scrollbar">
          <div className="p-3 sm:p-4">
            <div className="flex items-center justify-between mb-3 sm:mb-4">
              <span className="text-base sm:text-lg font-medium text-sidebar-primary-foreground">
                Advanced Settings
              </span>
              <button
                onClick={() =>
                  !isAwaitingResponse && setShowRightSidebar(false)
                }
                disabled={isAwaitingResponse}
                className={`p-1 rounded transition-colors ${
                  isAwaitingResponse
                    ? "cursor-not-allowed opacity-50"
                    : "hover:bg-sidebar-accent cursor-pointer"
                }`}
                title={
                  isAwaitingResponse
                    ? "Please wait for current operation to complete"
                    : "Hide right sidebar"
                }
              >
                <X className="w-4 h-4 cursor-pointer text-sidebar-primary-foreground" />
              </button>
            </div>

            {/* right sidebar accordion */}
            <div className="space-y-2">
              {/* Presets */}
              <div className="border-b border-sidebar-border">
                <button
                  onClick={() =>
                    !isAwaitingResponse && toggleAccordion("presets")
                  }
                  disabled={isAwaitingResponse}
                  className={`w-full text-left text-sm font-medium py-3 flex items-center justify-between transition-colors ${
                    isAwaitingResponse
                      ? "text-sidebar-primary-foreground/50 cursor-not-allowed"
                      : "text-sidebar-primary-foreground hover:text-sidebar-accent-foreground cursor-pointer"
                  }`}
                  title={
                    isAwaitingResponse
                      ? "Please wait for current operation to complete"
                      : "Toggle Vector Store settings"
                  }
                >
                  Vector Store
                  <ChevronDown
                    className={`w-4 h-4 transition-transform text-sidebar-primary-foreground ${
                      openAccordions.includes("presets") ? "rotate-180" : ""
                    }`}
                  />
                </button>
                {openAccordions.includes("presets") && (
                  <div className="pb-4">
                    <Select
                      value={preset}
                      onValueChange={(value) =>
                        !isAwaitingResponse && setPreset(value)
                      }
                      defaultValue="CFIR"
                      disabled={isAwaitingResponse}
                    >
                      <SelectTrigger
                        className={`bg-black-700 border-black ${
                          isAwaitingResponse
                            ? "opacity-50 cursor-not-allowed"
                            : ""
                        }`}
                      >
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-black border border-gray-700">
                        <SelectItem value="CFIR">CFIR</SelectItem>
                        <SelectItem value="Head and Neck">
                          Anatomical Regions - Head and Neck
                        </SelectItem>
                        <SelectItem value="Lower Extremity">
                          Anatomical Regions - Lower Extremity
                        </SelectItem>
                        <SelectItem value="Spine">
                          Anatomical Regions - Spine
                        </SelectItem>
                        <SelectItem value="Torso">
                          Anatomical Regions - Torso
                        </SelectItem>
                        <SelectItem value="Upper Extremity">
                          Anatomical Regions - Upper Extremity
                        </SelectItem>
                        <SelectItem value="Fractures">
                          Injury Typology - Fractures
                        </SelectItem>
                        <SelectItem value="Neurological Injuries">
                          Injury Typology - Neurological Injuries
                        </SelectItem>
                        <SelectItem value="Overuse or Chronic Injuries">
                          Injury Typology - Overuse/Chronic Injuries
                        </SelectItem>
                        <SelectItem value="Soft Tissue Injuries">
                          Injury Typology - Soft Tissue Injuries
                        </SelectItem>
                        <SelectItem value="Workplace or Repetitive Strain">
                          Injury Typology - Workplace/Repetitive Strain
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                )}
              </div>

              {/* Sampling */}
              <div className="border-b border-sidebar-border">
                <button
                  onClick={() =>
                    !isAwaitingResponse && toggleAccordion("sampling")
                  }
                  disabled={isAwaitingResponse}
                  className={`w-full text-left text-sm font-medium py-3 flex items-center justify-between transition-colors ${
                    isAwaitingResponse
                      ? "text-sidebar-primary-foreground/50 cursor-not-allowed"
                      : "text-sidebar-primary-foreground hover:text-sidebar-accent-foreground cursor-pointer"
                  }`}
                  title={
                    isAwaitingResponse
                      ? "Please wait for current operation to complete"
                      : "Toggle Sampling settings"
                  }
                >
                  Sampling
                  <ChevronDown
                    className={`w-4 h-4 transition-transform text-sidebar-primary-foreground ${
                      openAccordions.includes("sampling") ? "rotate-180" : ""
                    }`}
                  />
                </button>
                {openAccordions.includes("sampling") && (
                  <div className="pb-4">
                    <div className="space-y-4">
                      <div>
                        <label className="text-sm text-sidebar-primary-foreground block mb-2">
                          Temperature
                        </label>
                        <input
                          type="number"
                          value={temperature}
                          onChange={(e) =>
                            !isAwaitingResponse &&
                            setTemperature(Number(e.target.value))
                          }
                          step="0.1"
                          min="0"
                          max="2"
                          disabled={isAwaitingResponse}
                          className={`w-full border px-3 py-2 rounded ${
                            isAwaitingResponse
                              ? "bg-gray-500 border-gray-600 text-gray-400 cursor-not-allowed"
                              : "bg-black-700 border-sidebar-border text-sidebar-primary-foreground"
                          }`}
                          title={
                            isAwaitingResponse
                              ? "Please wait for current operation to complete"
                              : "Adjust temperature (0.0 - 2.0)"
                          }
                        />
                      </div>
                      <div>
                        <label className="text-sm text-sidebar-primary-foreground block mb-2">
                          Top P
                        </label>
                        <input
                          type="number"
                          value={topP}
                          onChange={(e) =>
                            !isAwaitingResponse &&
                            setTopP(Number(e.target.value))
                          }
                          step="0.1"
                          min="0"
                          max="1"
                          disabled={isAwaitingResponse}
                          className={`w-full border px-3 py-2 rounded ${
                            isAwaitingResponse
                              ? "bg-gray-500 border-gray-600 text-gray-400 cursor-not-allowed"
                              : "bg-black-700 border-sidebar-border text-sidebar-primary-foreground"
                          }`}
                          title={
                            isAwaitingResponse
                              ? "Please wait for current operation to complete"
                              : "Adjust top-p (0.0 - 1.0)"
                          }
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Strategies */}
              <div className="border-b border-sidebar-border">
                <button
                  onClick={() =>
                    !isAwaitingResponse && toggleAccordion("strategies")
                  }
                  disabled={isAwaitingResponse}
                  className={`w-full text-left text-sm font-medium py-3 flex items-center justify-between transition-colors ${
                    isAwaitingResponse
                      ? "text-sidebar-primary-foreground/50 cursor-not-allowed"
                      : "text-sidebar-primary-foreground hover:text-sidebar-accent-foreground cursor-pointer"
                  }`}
                  title={
                    isAwaitingResponse
                      ? "Please wait for current operation to complete"
                      : "Toggle Strategies settings"
                  }
                >
                  Strategies
                  <ChevronDown
                    className={`w-4 h-4 transition-transform text-sidebar-primary-foreground ${
                      openAccordions.includes("strategies") ? "rotate-180" : ""
                    }`}
                  />
                </button>
                {openAccordions.includes("strategies") && (
                  <div className="pb-4">
                    <Select
                      value={strategy}
                      onValueChange={(value) =>
                        !isAwaitingResponse && setStrategy(value)
                      }
                      defaultValue="no-workflow"
                      disabled={isAwaitingResponse}
                    >
                      <SelectTrigger
                        className={`bg-black-700 border-black text-sidebar-primary-foreground ${
                          isAwaitingResponse
                            ? "opacity-50 cursor-not-allowed"
                            : ""
                        }`}
                      >
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-black border border-gray-700">
                        <SelectItem value="no-workflow">No workflow</SelectItem>
                        <SelectItem value="ms-query-engine">
                          Multi-Step Query Engine
                        </SelectItem>
                        <SelectItem value="ms-reflection">
                          Multi-Strategy with Reflection
                        </SelectItem>
                        <SelectItem value="query-planning">
                          Query Planning
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                )}
              </div>

              {/* Mechanistic Interpretability */}
              <div className="border-b border-sidebar-border">
                <button
                  onClick={() =>
                    !isAwaitingResponse && toggleAccordion("interpretability")
                  }
                  disabled={isAwaitingResponse}
                  className={`w-full text-left text-sm font-medium py-3 flex items-center justify-between transition-colors ${
                    isAwaitingResponse
                      ? "text-sidebar-primary-foreground/50 cursor-not-allowed"
                      : "text-sidebar-primary-foreground hover:text-sidebar-accent-foreground cursor-pointer"
                  }`}
                  title={
                    isAwaitingResponse
                      ? "Please wait for current operation to complete"
                      : "Toggle Mechanistic Interpretability settings"
                  }
                >
                  <div className="flex items-center gap-2">
                    Mechanistic Interpretability
                  </div>
                  <ChevronDown
                    className={`w-4 h-4 transition-transform text-sidebar-primary-foreground ${
                      openAccordions.includes("interpretability")
                        ? "rotate-180"
                        : ""
                    }`}
                  />
                </button>
                {openAccordions.includes("interpretability") && (
                  <div className="pb-4">
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <label className="text-sm text-sidebar-primary-foreground">
                          Enable Interpretability
                        </label>
                        <button
                          onClick={() =>
                            !isAwaitingResponse &&
                            setMechanisticInterpretability(
                              !mechanisticInterpretability
                            )
                          }
                          disabled={isAwaitingResponse}
                          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                            mechanisticInterpretability
                              ? "bg-primary"
                              : isAwaitingResponse
                              ? "bg-gray-500"
                              : "bg-sidebar-border"
                          }`}
                          title={
                            isAwaitingResponse
                              ? "Please wait for current operation to complete"
                              : "Toggle mechanistic interpretability"
                          }
                        >
                          <span
                            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                              mechanisticInterpretability
                                ? "translate-x-6"
                                : "translate-x-1"
                            }`}
                          />
                        </button>
                      </div>
                      {mechanisticInterpretability && (
                        <div className="space-y-3">
                          <div>
                            <label className="text-sm text-sidebar-primary-foreground block mb-2">
                              Interpretability Level
                            </label>
                            <Select
                              defaultValue="basic"
                              disabled={isAwaitingResponse}
                            >
                              <SelectTrigger
                                className={`bg-black-700 border-black text-sidebar-primary-foreground ${
                                  isAwaitingResponse
                                    ? "opacity-50 cursor-not-allowed"
                                    : ""
                                }`}
                              >
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent className="bg-black border border-gray-700">
                                <SelectItem value="basic">Basic</SelectItem>
                                <SelectItem value="intermediate">
                                  Intermediate
                                </SelectItem>
                                <SelectItem value="advanced">
                                  Advanced
                                </SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                          <div className="text-xs text-sidebar-primary-foreground/70">
                            This feature will provide insights into how the AI
                            model processes and generates responses.
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* rate-limit modal*/}
      {showLimitModal && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/70">
          <div className="bg-card text-foreground rounded-lg shadow-lg max-w-sm w-full mx-4 p-6">
            <h3 className="text-lg font-semibold mb-2">Rate limit reached</h3>
            <p className="text-sm text-muted-foreground">
              Your message limit has been exceeded, sign up for more messages or
              try again in an hour
            </p>
            <div className="mt-4 flex justify-end space-x-2">
              <button
                onClick={() => router.push("/auth/signup")}
                disabled={isAwaitingResponse}
                className={`px-4 py-2 rounded ${
                  isAwaitingResponse
                    ? "bg-gray-500 cursor-not-allowed"
                    : "bg-primary hover:bg-primary/90 cursor-pointer"
                }`}
                title={
                  isAwaitingResponse
                    ? "Please wait for current operation to complete"
                    : "Sign up for more messages"
                }
              >
                Sign up
              </button>
              <button
                onClick={() => setShowLimitModal(false)}
                disabled={isAwaitingResponse}
                className={`px-4 py-2 border border-border rounded ${
                  isAwaitingResponse
                    ? "cursor-not-allowed"
                    : "hover:bg-accent cursor-pointer"
                }`}
                title={
                  isAwaitingResponse
                    ? "Please wait for current operation to complete"
                    : "Close this dialog"
                }
              >
                Later
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete confirmation modal */}
      {deletingId && (
        <div className="fixed inset-0 z-30 flex items-center justify-center bg-black/50">
          <div className="bg-gray-800 rounded-lg shadow-lg max-w-sm w-full mx-4">
            <div className="px-6 py-4 border-b border-gray-700">
              <h3 className="text-lg font-semibold text-white">
                Delete Conversation
              </h3>
            </div>
            <div className="px-6 py-4">
              <p className="text-sm text-gray-300">
                Are you sure you want to delete this conversation? This action
                cannot be undone.
              </p>
            </div>
            <div className="px-6 py-4 flex justify-end gap-2 border-t border-gray-700">
              <button
                onClick={() => setDeletingId(null)}
                disabled={isAwaitingResponse}
                className={`px-4 py-1 text-sm font-medium rounded ${
                  isAwaitingResponse
                    ? "text-gray-500 cursor-not-allowed"
                    : "text-gray-300 hover:text-white hover:bg-gray-700 cursor-pointer"
                }`}
                title={
                  isAwaitingResponse
                    ? "Please wait for current operation to complete"
                    : "Cancel deletion"
                }
              >
                Cancel
              </button>
              <button
                onClick={() => !isAwaitingResponse && deleteChat(deletingId)}
                disabled={isAwaitingResponse}
                className={`px-4 py-1 text-sm font-medium rounded ${
                  isAwaitingResponse
                    ? "text-gray-500 cursor-not-allowed"
                    : "text-white bg-red-600 hover:bg-red-700 cursor-pointer"
                }`}
                title={
                  isAwaitingResponse
                    ? "Please wait for current operation to complete"
                    : "Confirm deletion"
                }
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* search chats modal*/}
      {showSearchModal && (
        <div className="fixed inset-0 z-30 flex items-center justify-center bg-black/50">
          <div className="bg-gray-800 rounded-lg shadow-lg w-full max-w-md mx-4 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-foreground">
                Search Chats
              </h3>
              <button
                onClick={() => {
                  if (!isAwaitingResponse) {
                    setShowSearchModal(false);
                    setSearchQuery("");
                  }
                }}
                disabled={isAwaitingResponse}
                className={`p-1 rounded ${
                  isAwaitingResponse
                    ? "text-gray-500 cursor-not-allowed"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent cursor-pointer"
                }`}
                title={
                  isAwaitingResponse
                    ? "Please wait for current operation to complete"
                    : "Close search"
                }
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <input
              type="text"
              autoFocus
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder={
                isAwaitingResponse ? "Please wait..." : "Type to search..."
              }
              disabled={isAwaitingResponse}
              className={`w-full bg-input border border-border px-3 py-2 rounded mb-4 focus:outline-none focus:ring-2 focus:ring-primary ${
                isAwaitingResponse
                  ? "text-gray-400 cursor-not-allowed"
                  : "text-foreground"
              }`}
            />
            <div className="max-h-60 overflow-y-auto">
              {filteredResults.length === 0 ? (
                <div className="text-muted-foreground text-sm italic">
                  No matches found
                </div>
              ) : (
                filteredResults.map((conv) => {
                  const idx = conversations.findIndex((c) => c.id === conv.id);
                  return (
                    <div
                      key={conv.id}
                      onClick={() =>
                        !isAwaitingResponse && handleSearchResultClick(idx)
                      }
                      className={`px-3 py-2 rounded transition-colors truncate ${
                        isAwaitingResponse
                          ? "text-gray-500 cursor-not-allowed"
                          : "cursor-pointer hover:bg-accent text-foreground"
                      }`}
                      title={
                        isAwaitingResponse
                          ? "Please wait for current operation to complete"
                          : conv.title
                      }
                    >
                      {conv.title}
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
