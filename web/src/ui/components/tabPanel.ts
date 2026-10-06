/** Props for a tab panel that pairs with <Tabs idBase=...>. */
export function tabPanelProps(idBase: string, id: string) {
  return {
    role: 'tabpanel',
    id: `${idBase}-panel-${id}`,
    'aria-labelledby': `${idBase}-tab-${id}`,
    tabIndex: 0,
  } as const;
}
