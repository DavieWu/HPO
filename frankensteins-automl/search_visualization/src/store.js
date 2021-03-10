import Vue from 'vue';
import Vuex from 'vuex';
import { createSharedMutations } from 'vuex-electron';
import { DataSet } from 'vis-network';

Vue.use(Vuex);

export default new Vuex.Store({
  plugins: [
    createSharedMutations(),
  ],
  state: {
    nodes: new DataSet(),
    edges: new DataSet(),
  },
  getters: {
    nodes: state => state.nodes,
    edges: state => state.edges,
  },
  mutations: {
    addEvent: (state, event) => {
      if (event.event_type === 'NEW_NODE') {
        let color = 'blue';
        let nodeLabel = event.id.split('-')[0];
        if (event.node_type === 'root') {
          color = 'green';
        } else if (event.node_type === 'optimizer') {
          color = 'pink';
          nodeLabel = '0.000';
        }
        state.nodes.add({
          id: event.id,
          label: nodeLabel,
          color,
        });
        if (event.predecessor) {
          let edgeLabel = '';
          if (event.specified_interface) {
            edgeLabel = event.specified_interface;
          }
          state.edges.add({
            id: `${event.predecessor}-${event.id}`,
            from: event.predecessor,
            to: event.id,
            width: 1,
            label: edgeLabel,
          });
        }
      } else if (event.event_type === 'WEIGHT_UPDATE') {
        const width = Math.round(Math.max(1, (event.weight * 2)));
        state.edges.update({
          id: `${event.from}-${event.to}`,
          width,
        });
      } else if (event.event_type === 'OPTIMIZER_UPDATE') {
        let color = 'pink';
        if (event.score > 0) {
          color = 'red';
        }
        state.nodes.update({
          id: event.id,
          label: event.score.toFixed(3),
          color,
        });
      }
    },
  },
  actions: {
    ADD_EVENT_ACTION: ({ commit }, payload) => {
      commit('addEvent', payload.event);
    },
  },
});
