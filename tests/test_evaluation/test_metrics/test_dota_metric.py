# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import unittest

import numpy as np
import torch

from mmrotate.evaluation import DOTAMetric


class TestDOTAMetric(unittest.TestCase):

    def _create_dummy_data_sample(self):
        bboxes = np.array([[23, 31, 10.0, 20.0, 0.0],
                           [100, 120, 10.0, 20.0, 0.1],
                           [150, 160, 10.0, 20.0, 0.2],
                           [250, 260, 10.0, 20.0, 0.3]])
        labels = np.array([0] * 4)
        bboxes_ignore = np.array([[0] * 5])
        labels_ignore = np.array([0])
        pred_bboxes = np.array([[23, 31, 10.0, 20.0, 0.0],
                                [100, 120, 10.0, 20.0, 0.1],
                                [150, 160, 10.0, 20.0, 0.2],
                                [250, 260, 10.0, 20.0, 0.3]])
        pred_scores = np.array([1.0, 0.98, 0.96, 0.95])
        pred_labels = np.array([0, 0, 0, 0])
        return [
            dict(
                img_id='P2805__1024__0___0',
                gt_instances=dict(
                    bboxes=torch.from_numpy(bboxes),
                    labels=torch.from_numpy(labels)),
                ignored_instances=dict(
                    bboxes=torch.from_numpy(bboxes_ignore),
                    labels=torch.from_numpy(labels_ignore)),
                pred_instances=dict(
                    bboxes=torch.from_numpy(pred_bboxes),
                    scores=torch.from_numpy(pred_scores),
                    labels=torch.from_numpy(pred_labels)))
        ]

    def _create_dummy_data_sample_qbox(self):
        bboxes = np.array([[13, 11, 13, 31, 23, 31, 23, 11],
                           [90, 100, 110, 100, 110, 140, 90, 140],
                           [140, 140, 150, 140, 150, 160, 140, 160],
                           [240, 250, 250, 250, 250, 260, 240, 260]])
        labels = np.array([0] * 4)
        bboxes_ignore = np.array([[0] * 8])
        labels_ignore = np.array([0])
        pred_bboxes = np.array([[13, 11, 13, 31, 23, 31, 23, 11],
                                [90, 100, 110, 100, 110, 140, 90, 140],
                                [140, 140, 150, 140, 150, 160, 140, 160],
                                [240, 250, 250, 250, 250, 260, 240, 260]])
        pred_scores = np.array([1.0, 0.98, 0.96, 0.95])
        pred_labels = np.array([0, 0, 0, 0])
        return [
            dict(
                img_id='P2805__1024__0___0',
                gt_instances=dict(
                    bboxes=torch.from_numpy(bboxes),
                    labels=torch.from_numpy(labels)),
                ignored_instances=dict(
                    bboxes=torch.from_numpy(bboxes_ignore),
                    labels=torch.from_numpy(labels_ignore)),
                pred_instances=dict(
                    bboxes=torch.from_numpy(pred_bboxes),
                    scores=torch.from_numpy(pred_scores),
                    labels=torch.from_numpy(pred_labels)))
        ]

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_init(self):
        # test invalid iou_thrs
        with self.assertRaises(AssertionError):
            DOTAMetric(iou_thrs={'a', 0.5})

        metric = DOTAMetric(iou_thrs=0.6)
        self.assertEqual(metric.iou_thrs, [0.6])

    def test_eval(self):
        metric = DOTAMetric()
        metric.dataset_meta = {'classes': ('plane', )}
        metric.process({}, self._create_dummy_data_sample())
        results = metric.evaluate(size=1)
        targets = {'dota/AP50': 1.0, 'dota/mAP': 1.0}
        self.assertDictEqual(results, targets)

    def test_process_restores_gt_to_prediction_coordinate_space(self):
        """Mixed-resolution test data must be evaluated in one space."""
        metric = DOTAMetric()
        metric.dataset_meta = {'classes': ('pipe', )}
        original_box = torch.tensor([[400.0, 300.0, 20.0, 500.0, 0.0]])
        resized_box = torch.tensor(
            [[368.0, 256.0, 18.4, 512.0 / 600 * 500, 0.0]])
        sample = dict(
            img_id='nonuniform_resize_fixture',
            scale_factor=(736 / 800, 512 / 600),
            gt_instances=dict(
                bboxes=resized_box,
                labels=torch.tensor([0])),
            ignored_instances=dict(
                bboxes=torch.empty((0, 5)),
                labels=torch.empty((0,), dtype=torch.long)),
            # SingleStageDetector.predict returns this space with rescale=True.
            pred_instances=dict(
                bboxes=original_box,
                scores=torch.tensor([1.0]),
                labels=torch.tensor([0])))

        metric.process({}, [sample])
        restored = metric.results[0][0]['bboxes']
        np.testing.assert_allclose(restored, original_box.numpy(), atol=1e-4)

        # test multi-threshold
        metric = DOTAMetric(iou_thrs=[0.1, 0.5])
        metric.dataset_meta = dict(classes=('plane', ))
        metric.process({}, self._create_dummy_data_sample())
        results = metric.evaluate(size=1)
        targets = {'dota/AP10': 1.0, 'dota/AP50': 1.0, 'dota/mAP': 1.0}
        self.assertDictEqual(results, targets)

    def test_format(self):
        metric = DOTAMetric()
        with self.assertRaises(AssertionError):
            DOTAMetric(format_only=True, outfile_prefix=None)

        metric = DOTAMetric(
            format_only=True, outfile_prefix=f'{self.tmp_dir.name}/test')
        metric.dataset_meta = dict(classes=('plane', ))
        metric.process({}, self._create_dummy_data_sample())
        eval_results = metric.evaluate(size=1)
        self.assertDictEqual(eval_results, dict())
        self.assertTrue(osp.exists(f'{self.tmp_dir.name}/test.bbox.json'))

    def test_merge_patches(self):
        metric = DOTAMetric(
            format_only=True,
            merge_patches=True,
            outfile_prefix=f'{self.tmp_dir.name}/Task1')
        metric.dataset_meta = dict(classes=('plane', ))
        metric.process({}, self._create_dummy_data_sample())
        eval_results = metric.evaluate(size=1)
        self.assertDictEqual(eval_results, dict())
        self.assertTrue(osp.exists(f'{self.tmp_dir.name}/Task1/Task1.zip'))

    def test_eval_qbox(self):
        metric = DOTAMetric(predict_box_type='qbox')
        metric.dataset_meta = {'classes': ('plane', )}
        metric.process({}, self._create_dummy_data_sample_qbox())
        results = metric.evaluate(size=1)
        targets = {'dota/AP50': 1.0, 'dota/mAP': 1.0}
        self.assertDictEqual(results, targets)

    def test_format_qbox(self):
        metric = DOTAMetric(predict_box_type='qbox')
        with self.assertRaises(AssertionError):
            DOTAMetric(format_only=True, outfile_prefix=None)

        metric = DOTAMetric(
            predict_box_type='qbox',
            format_only=True,
            outfile_prefix=f'{self.tmp_dir.name}/test')
        metric.dataset_meta = dict(classes=('plane', ))
        metric.process({}, self._create_dummy_data_sample_qbox())
        eval_results = metric.evaluate(size=1)
        self.assertDictEqual(eval_results, dict())
        self.assertTrue(osp.exists(f'{self.tmp_dir.name}/test.bbox.json'))

    def test_merge_patches_qbox(self):
        metric = DOTAMetric(
            predict_box_type='qbox',
            format_only=True,
            merge_patches=True,
            outfile_prefix=f'{self.tmp_dir.name}/Task1')
        metric.dataset_meta = dict(classes=('plane', ))
        metric.process({}, self._create_dummy_data_sample_qbox())
        eval_results = metric.evaluate(size=1)
        self.assertDictEqual(eval_results, dict())
        self.assertTrue(osp.exists(f'{self.tmp_dir.name}/Task1/Task1.zip'))
