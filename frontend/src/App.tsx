import { useEffect, useState } from "react";
import ProjectBrowser from "./components/ProjectBrowser";
import Editor from "./components/Editor";
import { ModelsDialog } from "./components/Dialogs";
import { api } from "./lib/api";

export default function App() {
  const [projectId, setProjectId] = useState<string | null>(() => new URLSearchParams(location.search).get("scene"));
  const [models, setModels] = useState(false);
  const [testMode, setTestMode] = useState(false);

  useEffect(() => { api.health().then((h) => setTestMode(h.test_mode)).catch(() => {}); }, []);
  useEffect(() => {
    const url = new URL(location.href);
    if (projectId) url.searchParams.set("scene", projectId); else url.searchParams.delete("scene");
    history.replaceState(null, "", url);
  }, [projectId]);

  return (
    <>
      {testMode && (
        <div role="alert" style={{ position: "fixed", top: 4, left: "50%", transform: "translateX(-50%)", zIndex: 70, background: "#5a1f1b", color: "#ffd6d2", padding: "3px 12px", borderRadius: 3, fontSize: 11 }}>
          Test mode (SPATIALAI_MOCK_MODELS=1): neural networks are replaced by stand-ins
        </div>
      )}
      {projectId
        ? <Editor key={projectId} projectId={projectId} onClose={() => setProjectId(null)} onOpenProject={setProjectId} />
        : <ProjectBrowser onOpen={setProjectId} onShowModels={() => setModels(true)} />}
      {models && <ModelsDialog onClose={() => setModels(false)} />}
    </>
  );
}
