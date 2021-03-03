'use strict';

const router = require('../../ivis-core/server/lib/router-async').create();
const log = require('../../ivis-core/server/lib/log');
const passport = require('../../ivis-core/server/lib/passport');
const axios = require('axios');
const config = require('../../ivis-core/server/lib/config');

function getEnforcerUrl(path) {
    return config.enforcer.url + path;
}

router.getAsync('/access/:workerId', async (req, res) => {
    let workerId = req.params.workerId;
    if (workerId === '221') {
        workerId = 'A-worker-001';
    }

    const resp = await axios.post(getEnforcerUrl('access/' + workerId));
    return res.json(resp.data);
});

router.getAsync('/validate/:subjectId/:verb/:objectId', async (req, res) => {
    const resp = await axios.post(getEnforcerUrl(`validate/${req.params.subjectId}/${req.params.verb}/${req.params.objectId}`));
    return res.json(resp.data);
});

// subjectId = A-foreman    verb = read(phoneNo)                  objectId = A-worker-001
// subjectId = A-foreman    verb = read(aggregatedTemperature)    objectId = machine-A


module.exports = router;
