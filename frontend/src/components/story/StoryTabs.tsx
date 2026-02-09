import { Tab } from '@headlessui/react';
import { MachineLearningStory } from './MachineLearningStory';
import { FrontendStory } from './FrontendStory';
import { BackendStory } from './BackendStory';
import { DeploymentStory } from './DeploymentStory';

export function StoryTabs() {
  const tabs = [
    { name: 'Machine Learning', component: MachineLearningStory },
    { name: 'Frontend', component: FrontendStory },
    { name: 'Backend', component: BackendStory },
    { name: 'Deployment', component: DeploymentStory },
  ];

  return (
    <Tab.Group>
      <Tab.List className="story-tabs">
        {tabs.map((tab) => (
          <Tab key={tab.name} className="story-tab">
            {({ selected }: { selected: boolean }) => (
              <button className={selected ? 'tab-active' : 'tab-inactive'}>
                {tab.name}
              </button>
            )}
          </Tab>
        ))}
      </Tab.List>
      <Tab.Panels className="story-panels">
        {tabs.map((tab) => {
          const Component = tab.component;
          return (
            <Tab.Panel key={tab.name} className="story-panel">
              <Component />
            </Tab.Panel>
          );
        })}
      </Tab.Panels>
    </Tab.Group>
  );
}
