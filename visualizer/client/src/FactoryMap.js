'use strict';

import React, {Component} from "react";
import {withComponentMixins} from "../../ivis-core/client/src/lib/decorator-helpers";
import {withTranslation} from "../../ivis-core/client/src/lib/i18n";
import {withAsyncErrorHandler, withErrorHandling} from "../../ivis-core/client/src/lib/error-handling";
import {withPageHelpers} from "../../ivis-core/client/src/lib/page-common";
import {requiresAuthenticatedUser} from "../../ivis-core/client/src/lib/page";
import {Panel} from "../../ivis-core/client/src/lib/panel";
import {Button} from "../../ivis-core/client/src/lib/bootstrap-components";
import styles from "./FactoryMap.scss";
import axios from "../../ivis-core/client/src/lib/axios";
import {getUrl} from "../../ivis-core/client/src/lib/urls";
import moment from "moment";
import {SVG} from "../../ivis-core/client/src/ivis/SVG";
import factorySvg from "../images/factory.svg";
import { select, event as d3Event, mouse } from 'd3-selection';

const State = {
    // This has to be aligned with enforcer/src/main/scala/trust40/k4case/Timeline.scala
    START: 0,
    PLAYING: 1,
    PAUSED: 2,
    END: 3
};

const SwipeCardState = {
    IDLE: 'info',
    ALLOWED: 'success',
    NOT_ALLOWED: 'danger'
};

const refreshInterval = 50;
const minorStepsInRefetchPeriod = 5;

const AccessResultToSwipeCardStateMapping = {
    'none': SwipeCardState.IDLE,
    'allowed': SwipeCardState.ALLOWED,
    'not allowed': SwipeCardState.NOT_ALLOWED
};

@withComponentMixins([
    withTranslation,
    withErrorHandling,
    withPageHelpers,
    requiresAuthenticatedUser
])export default class FactoryMap extends Component {
    constructor(props) {
        super(props);

        this.state = {
            playState: State.START,
            swipeCardState: SwipeCardState.IDLE,
            ts: null,
            workers: [],
            selectedWorker: null
        };

        this.swipeCardTimeout = null;

        this.reset();

        this.refreshIntervalId = null;
    }

    reset() {
        this.keyframes = [];
        this.minorStep = 0;
        this.initialKeyframesFetched = false;
    }

    @withAsyncErrorHandler
    async play() {
        await axios.post(getUrl('sim/play'));

        this.setState({
            playState: State.PLAYING
        });
    }

    @withAsyncErrorHandler
    async stop() {
        await axios.post(getUrl('sim/reset'));

        this.setState({
            playState: State.START,
            ts: null
        });

        this.reset();
    }

    @withAsyncErrorHandler
    async pause() {
        await axios.post(getUrl('sim/pause'));

        this.setState({
            playState: State.PAUSED
        });
    }

    @withAsyncErrorHandler
    async swipeCard(selWorkerId) {
        const resp = await axios.post(getUrl('sim/access/' + selWorkerId));

        this.setState({
            swipeCardState: AccessResultToSwipeCardStateMapping[resp.data.result]
        });

        clearTimeout(this.swipeCardTimeout);
        this.swipeCardTimeout = setTimeout(() => {
            this.setState({
                swipeCardState: SwipeCardState.IDLE
            });
        }, 1000)
    }

    @withAsyncErrorHandler
    async getStatus() {
        const resp = await axios.get(getUrl('sim/status'));
        const ts = moment(resp.data.time);
        const kf = this.keyframes;

        if (kf.length > 0 && kf[kf.length - 1].ts.isAfter(ts)) {
            // This may happen if we receive status from previous epoch after stop. The status from the previous
            // epoch will set ts to something high. The reset status that we receive afterwards won't match
            // the last ts, which is what we check here.
            this.reset();

        } else {
            kf.push({
                ts,
                workers: resp.data.workers,
                permissions: resp.data.permissions,
                rejectedPermissions: resp.data.rejectedPermissions
            });

            const playState = resp.data.playState;
            if (this.state.playState !== playState) {
                this.setState({
                    playState
                });
            }

            if (kf.length === 3 && !this.initialKeyframesFetched) {
                this.initialKeyframesFetched = true;
                this.minorStep = 0;
            }
        }
    }

    refresh() {
        if (!this.initialKeyframesFetched) {
            this.minorStep += 1;
            if (this.minorStep === minorStepsInRefetchPeriod) {
                this.minorStep = 0;
                // noinspection JSIgnoredPromiseFromCall
                this.getStatus();
            }

        } else {
            const kf = this.keyframes;

            if (kf.length >= 2) {
                const last = kf[0];
                const next = kf[1];
                const alpha = this.minorStep / minorStepsInRefetchPeriod;

                const interp = (last, next) => last * (1-alpha) + next * alpha;

                const ts = moment(interp(last.ts.valueOf(), next.ts.valueOf()));

                if (!this.state.ts || ts.valueOf() !== this.state.ts.valueOf()) {

                    const workers = [];
                    for (const key in last.workers) {
                        if (key in next.workers) {
                            const lastWorker = last.workers[key];
                            const nextWorker = next.workers[key];

                            let symbol;
                            if (key.endsWith('foreman')) {
                                symbol = 'foreman';
                            } else if (key === 'A-worker-001') {
                                symbol = 'user';
                            } else {
                                symbol = 'worker'
                            }

                            workers.push({
                                id: key,
                                symbol,
                                x: interp(lastWorker.position.x, nextWorker.position.x),
                                y: interp(lastWorker.position.y, nextWorker.position.y),
                                hasHeadGear: lastWorker.hasHeadGear,
                                standbyFor: lastWorker.standbyFor
                            });
                        }
                    }


                    this.setState({
                        ts,
                        workers,
                        permissions: last.permissions,
                        rejectedPermissions: last.rejectedPermissions
                    });
                }
            }

            this.minorStep += 1;
            if (this.minorStep === minorStepsInRefetchPeriod) {
                this.minorStep = 0;

                if (kf.length >= 2) {
                    kf.shift();
                }

                // noinspection JSIgnoredPromiseFromCall
                this.getStatus();
            }
        }
    }

    componentDidMount() {
        this.getStatus();
        this.refreshIntervalId = setInterval(::this.refresh, refreshInterval);
    }

    componentWillUnmount() {
        clearInterval(this.refreshIntervalId);
    }

    render() {
        const t = this.props.t;
        const playState = this.state.playState;

        const ts = this.state.ts;
        const tsFormatted = ts ? ts.utc().format('HH:mm:ss') : null;

        /*
        if (this.state.permissions && this.state.permissions.length > 0) {
            console.log(this.state.permissions);
        }
         */

        const handleHeaderRect = sigCid => {
            return node => node.style('fill', data[sigCid] > 0.01 ? 'lime' : 'red');
        };

        const handleTextGroup = (sigCidOn, offAsByPassed, labelsFun) => {
            return node => node.selectAll('text').each(function(d, i) {
                if (data[sigCidOn] > 0.01) {
                    const labels = labelsFun();
                    select(this).text(labels[i]);
                } else {
                    if (offAsByPassed) {
                        select(this).text(i === 2 ? 'BY-PASSED' : '');
                    } else {
                        select(this).text('');
                    }
                }
            });
        };

        const selWorkerId = this.state.selectedWorker;
        const selWorkerPerms = new Set();
        let workerDetails;
        if (selWorkerId) {
            let selWorker = this.state.workers.find(worker => worker.id === selWorkerId);

            const perms = [];
            for (const perm of this.state.permissions) {
                if (perm[0] === selWorkerId) {
                    perms.push(<span key={`${perm[0]}-${perm[1]}-${perm[2]}`} className={`badge badge-success ${styles.perm}`}>{perm[1]} {perm[2]}</span>);
                    selWorkerPerms.add(perm[1] + ' ' + perm[2]);
                }
            }

            const rejectedPerms = [];
            for (const perm of this.state.rejectedPermissions) {
                if (perm[0] === selWorkerId) {
                    const reasons = [];
                    for (const rejPerm of perm[3]) {
                        reasons.push(<span key={`${rejPerm[1]}-${rejPerm[2]}-${rejPerm[3]}`} className={`badge badge-warning ${styles.perm}`}>{rejPerm[1]} {rejPerm[3]} {rejPerm[2]}</span>);
                    }

                    rejectedPerms.push(<li key={`${perm[0]}-${perm[1]}-${perm[2]}`}><span className={`badge badge-danger ${styles.perm}`}>{perm[1]} {perm[2]}</span> <span className={styles.rejectedBy}>rejected by</span> {reasons}</li>);
                }

            }

            workerDetails = (
                <>
                    <div className={`card-body ${styles.detailsSection}`}>
                        <div className={`card-title ${styles.detailsSectionHeader}`}>{t('Identification')}</div>
                        <div className="card-text">
                            {selWorkerId}
                        </div>
                    </div>
                    <div className={`card-body ${styles.detailsSection}`}>
                        <div className={`card-title ${styles.detailsSectionHeader}`}>{t('Permissions')}</div>
                        <div className="card-text">
                            {perms.length > 0 ? perms : t('None')}
                        </div>
                    </div>

                    <div className={`card-body ${styles.detailsSection}`}>
                        <div className={`card-title ${styles.detailsSectionHeader}`}>{t('Denied Permissions (due to privacy rules)')}</div>
                        <div className="card-text">
                            {rejectedPerms.length > 0 ?
                                <ul className={styles.rejectedPerms}>
                                    {rejectedPerms}
                                </ul>
                            : t('None')
                            }
                        </div>
                    </div>

                    <div className={`card-body ${styles.detailsSection}`}>
                        <div className={`card-title ${styles.detailsSectionHeader}`}>{t('Has head gear')}</div>
                        <div className="card-text">
                            {selWorker && selWorker.hasHeadGear ? t('Yes') : t('No')}
                        </div>
                    </div>

                    {selWorker && selWorker.standbyFor &&
                    <div className={`card-body ${styles.detailsSection}`}>
                        <div className={`card-title ${styles.detailsSectionHeader}`}>{t('Standby for')}</div>
                        <div className="card-text">
                            {selWorker.standbyFor}
                        </div>
                    </div>
                    }
                </>
            );
        } else {
            workerDetails = (
                <div className={`card-body ${styles.detailsSection}`}>
                    <div className="card-text">
                        <div className={styles.help}>{t('Select a worker to see details')}</div>
                    </div>
                </div>
            );
        }

        const getPlaceColor = perm => {
            if (!selWorkerId) return 'white';
            else if (selWorkerPerms.has(perm)) return '#4dbd74';
            else return '#f86c6b';
        };

        return (
            <Panel title={t('Factory Map')}>
                <div className="row">
                    <div className="col-12 col-lg-9">
                        <div className="row">
                            <div className="col-12 col-lg-9 mb-3">
                                { playState === State.PLAYING &&
                                    <Button className={`btn-primary ${styles.controlButton}`} icon="pause" onClickAsync={::this.pause} />
                                }
                                { (playState === State.START || playState === State.PAUSED) &&
                                <Button className={`btn-primary ${styles.controlButton}`} icon="play" onClickAsync={::this.play} />
                                }
                                { (playState === State.END) &&
                                <Button className={`btn-primary ${styles.controlButton}`} icon="play" disabled={true} />
                                }
                                <Button className={`btn-danger ${styles.controlButton}`} icon="stop" onClickAsync={::this.stop} disabled={playState === State.START}/>
                                <span className={styles.timestamp}>{tsFormatted}</span>
                            </div>
                            <div className="col-12 col-lg-3 mb-3 text-lg-right">
                                {selWorkerId && <Button className={`btn-${this.state.swipeCardState} ${styles.controlButton}`} icon="id-card" label={t('Swipe card')} onClickAsync={async () => await this.swipeCard(selWorkerId)} />}
                            </div>
                        </div>
                        <div>
                            <SVG
                                source={factorySvg}
                                init={node => {
                                    const viewBox = node.viewBox.baseVal;
                                    const nodeSel = select(node);
                                    const workerSizeHalf = 8.66 / 2;

                                    nodeSel
                                        .append('rect')
                                        .attr('x', viewBox.x)
                                        .attr('y', viewBox.y)
                                        .attr('width', viewBox.width)
                                        .attr('height', viewBox.height)
                                        .attr('fill-opacity', 0)
                                        .attr('stroke', 'none')
                                        .on("click", d => {
                                            const mousePos = mouse(node);
                                            const x = mousePos[0];
                                            const y = mousePos[1];
                                            let selectedWorker = null;
                                            for (const worker of this.state.workers) {
                                                if (worker.x - workerSizeHalf <= x && worker.x + workerSizeHalf >= x && worker.y - workerSizeHalf <= y && worker.y + workerSizeHalf >= y) {
                                                    selectedWorker = worker.id;
                                                }
                                            }
                                            this.setState({selectedWorker});
                                        });

                                    nodeSel
                                        .select('#Positions')
                                        .attr('display', 'none');
                                }}
                                data={{
                                    MainGate: node => node.style('fill', getPlaceColor('enter factory')),
                                    Gate1: node => node.style('fill', getPlaceColor('enter workplace-A')),
                                    Gate2: node => node.style('fill', getPlaceColor('enter workplace-B')),
                                    Gate3: node => node.style('fill', getPlaceColor('enter workplace-C')),
                                    Dispenser: node => node.style('fill', getPlaceColor('use dispenser')),
                                    MachineA: node => node.style('fill', getPlaceColor('read(aggregatedTemperature) machine-A')),
                                    MachineB: node => node.style('fill', getPlaceColor('read(aggregatedTemperature) machine-B')),
                                    MachineC: node => node.style('fill', getPlaceColor('read(aggregatedTemperature) machine-C')),
                                    Workers: node => {
                                        const workers = this.state.workers;

                                        node.selectAll('use')
                                            .data(workers, d => d.id)
                                            .enter()
                                            .append('use');

                                        node.selectAll('use')
                                            .data(workers, d => d.id)
                                            .exit()
                                            .remove();

                                        node.selectAll('use')
                                            .data(workers, d => d.id)
                                            .attr('href', d => (d.id === this.state.selectedWorker ? "#selected-" : "#") + d.symbol)
                                            .attr('x', d => d.x - 10)
                                            .attr('y', d => d.y - 10);
                                    }
                                }}
                            />
                        </div>
                    </div>
                    <div className="col-12 col-lg-3">
                        <div className="card">
                            <div className="card-header">
                                <h4 className={styles.detailsHeader}>{t('Worker Details')}</h4>
                            </div>
                            {workerDetails}
                        </div>
                    </div>
                </div>
            </Panel>
        );
    }
}