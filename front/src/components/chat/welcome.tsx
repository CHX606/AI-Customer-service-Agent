import { useAui } from "@assistant-ui/react";
import { Prompts, Welcome as ChatWelcome } from "@ant-design/x";
import { Avatar } from "antd";
import { ArrowUpRight, Sparkles } from "lucide-react";
import type { TenantPublicProfile } from "../../types/profile";

export function Welcome({ profile }: { profile: TenantPublicProfile }) {
  const aui = useAui();
  return (
    <section className="welcome" aria-labelledby="welcome-title">
      <ChatWelcome variant="borderless"
        icon={<Avatar shape="square" size={56} icon={<Sparkles size={28} />} className="brand-avatar" />}
        title={<span id="welcome-title">{profile.welcome_title}</span>}
        description={profile.welcome_description}
        classNames={{ root: "welcome-heading", description: "welcome-description" }}
      />
      {profile.suggested_questions.length > 0 && <Prompts
        title="你可以这样问" wrap className="welcome-prompts"
        classNames={{ list: "suggestion-grid", item: "suggestion-item" }}
        items={profile.suggested_questions.map((prompt, index) => ({ key: String(index), description: prompt, icon: <ArrowUpRight size={18} /> }))}
        onItemClick={({ data }) => {
          aui.composer.setText(profile.suggested_questions[Number(data.key)]);
          aui.composer.send();
        }}
      />}
    </section>
  );
}
