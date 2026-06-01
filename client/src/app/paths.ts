export const inspectionModePaths = ["photo", "snapshot", "realtime"] as const;
export type InspectionModePath = (typeof inspectionModePaths)[number];

export const settingsSectionPaths = ["system"] as const;
export type SettingsSectionPath = (typeof settingsSectionPaths)[number];

export const paths = {
  home: () => "/",

  groups: () => "/groups",
  groupDetail: (groupId: string) => `/groups/${groupId}`,
  standardDetail: (groupId: string, standardId: string) => {
    return `/groups/${groupId}/standards/${standardId}`;
  },
  standardImage: (groupId: string, standardId: string, imageId: string) => {
    return `/groups/${groupId}/standards/${standardId}/images/${imageId}`;
  },

  training: () => "/training",
  trainingGroup: (groupId: string) => `/training/${groupId}`,
  trainingModel: (groupId: string, modelId: string) => `/training/${groupId}/models/${modelId}`,

  inspection: () => "/inspection",
  inspectionMode: (mode: InspectionModePath) => `/inspection/${mode}`,
  inspectionGroup: (mode: string, groupId: string) => {
    return `/inspection/${mode}/groups/${groupId}`;
  },
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
