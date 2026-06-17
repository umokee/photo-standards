export const inspectionModePaths = ["photo", "snapshot", "realtime"] as const;
export type InspectionModePath = (typeof inspectionModePaths)[number];

export const settingsSectionPaths = ["system"] as const;
export type SettingsSectionPath = (typeof settingsSectionPaths)[number];

export const paths = {
  home: () => "/",

  groups: () => "/groups",
  groupDetail: (groupId: string) => `/groups/${groupId}`,
  assetOverview: (groupId: string) => `/groups/${groupId}`,
  assetReferences: (groupId: string) => `/groups/${groupId}/references`,
  assetClasses: (groupId: string) => `/groups/${groupId}/classes`,
  groupReferences: (groupId: string) => `/groups/${groupId}/references`,
  groupClasses: (groupId: string) => `/groups/${groupId}/classes`,
  standardDetail: (groupId: string, standardId: string) => {
    return `/groups/${groupId}/standards/${standardId}`;
  },
  standardImage: (groupId: string, standardId: string, imageId: string) => {
    return `/groups/${groupId}/standards/${standardId}/images/${imageId}`;
  },

  training: () => "/training",
  trainingGroup: (groupId: string) => `/training/${groupId}`,
  trainingOverview: (groupId: string) => `/training/${groupId}`,
  trainingModels: (groupId: string) => `/training/${groupId}/models`,
  trainingRuns: (groupId: string) => `/training/${groupId}/runs`,
  trainingModel: (groupId: string, modelId: string) => `/training/${groupId}/models/${modelId}`,

  inspection: () => "/inspection",
  inspectionMode: (mode: InspectionModePath) => `/inspection/${mode}`,
  inspectionGroup: (mode: string, groupId: string) => {
    return `/inspection/${mode}/groups/${groupId}`;
  },
  inspectionGroupRuns: (groupId: string) => `/inspection-history/${groupId}`,
  inspectionStandard: (mode: string, groupId: string, standardId: string) => {
    return `/inspection/${mode}/groups/${groupId}/standards/${standardId}`;
  },

  inspectionHistory: () => "/inspection-history",
  inspectionHistoryGroup: (groupId: string) => {
    return `/inspection-history/${groupId}`;
  },
  inspectionHistoryDetail: (groupId: string, inspectionId: string) => {
    return `/inspection-history/${groupId}/${inspectionId}`;
  },

  cameras: () => "/cameras",
  cameraDetail: (cameraId: string) => `/cameras/${cameraId}`,

  settings: () => "/settings",
  settingsSection: (section: SettingsSectionPath) => `/settings/${section}`,
} as const;
