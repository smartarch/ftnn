'use strict';

import React from "react";

import em from '../../ivis-core/client/src/lib/extension-manager';
import FactoryMap from './FactoryMap';
import {NavLink} from "../../ivis-core/client/src/lib/page";

import ivisConfig from "ivisConfig";

em.set('app.title', 'Trust4.0 Demo');

em.on('client.installRoutes', (structure, t) => {
    structure.children['workspaces'].children['factory-map'] = {
        title: t('Factory Map'),
        link: '/workspaces/factory-map',
        panelComponent: FactoryMap,
        secondaryMenuComponent: null
    };

    structure.link = () => ivisConfig.isAuthenticated ? '/workspaces/factory-map' : '/login';
});

em.on('client.mainMenuAuthenticated.installWorkspaces', (workspaces, t) => {
    workspaces.push(<NavLink key="factory-map" to="/workspaces/factory-map">{t('Factory Map')}</NavLink>);
});

require('../../ivis-core/client/src/root-trusted');

