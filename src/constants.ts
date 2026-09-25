/**
 * Central navigation registry — LoanLens (Loan Default Prediction & Risk Intelligence).
 * Strictly focused on Loan Default Prediction & Risk Assessment.
 */

export interface NavItem {
  key: string;
  label: string;
  route: string;
  icon: string;
  status: 'live' | 'soon';
  badge?: string;
}

export interface NavGroup {
  group: string;
  items: NavItem[];
}

export const NAV: NavGroup[] = [
  {
    group: 'WORKSPACE',
    items: [
      { key: 'overview', label: 'Overview', route: '/overview', icon: 'grid', status: 'live' },
      { key: 'assess', label: 'New Assessment', route: '/assess', icon: 'target', status: 'live' },
      { key: 'context', label: 'Model Context', route: '/model/context', icon: 'git-branch', status: 'live' },
      { key: 'comparison', label: 'Model Comparison', route: '/model/comparison', icon: 'sliders', status: 'live' },
      { key: 'insights', label: 'Data Insights', route: '/data/insights', icon: 'database', status: 'live' },
      { key: 'details', label: 'Model Details', route: '/model/details', icon: 'cpu', status: 'live' },
    ],
  },
  {
    group: 'PORTFOLIO',
    items: [
      { key: 'history', label: 'Assessment History', route: '/predictions', icon: 'clock', status: 'live' },
      { key: 'simulator', label: 'Sensitivity Simulator', route: '/simulator', icon: 'activity', status: 'live' },
    ],
  },
];
