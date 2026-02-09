import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { StoryTabs } from '../story/StoryTabs';
import { PredictionForm } from '../form/PredictionForm';

export function SplitLayout() {
  return (
    <PanelGroup direction="horizontal" className="split-layout">
      {/* Left Panel - Story/Documentation */}
      <Panel defaultSize={50} minSize={30} className="left-panel">
        <div className="panel-content">
          <StoryTabs />
        </div>
      </Panel>

      {/* Resize Handle */}
      <PanelResizeHandle className="resize-handle" />

      {/* Right Panel - Prediction Form */}
      <Panel defaultSize={50} minSize={30} className="right-panel">
        <div className="panel-content">
          <PredictionForm />
        </div>
      </Panel>
    </PanelGroup>
  );
}
