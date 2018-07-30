""" File to store all Dataset editor functions """


def delete_image(self, image_id):
    """ Deletes the image associated with the image_id """
    for ann_id in self.img_to_ann[image_id]:
        del self.ann_infos[ann_id]
    del self.img_to_ann[image_id]
    del self.image_infos[image_id]
    self.all_image_index = self.list_all_image_index()

def edit_image_path(self, image_id, new_image_path):
    """ Edits the image path of an entry """
    self.image_infos[image_id]['image_path'] = new_image_path

def edit_image_url(self, image_id, new_image_url):
    """ Edits the image url of an entry """
    self.image_infos[image_id]['image_url'] = new_image_url

def edit_ann_img_id(self, ann_id, new_image_id):
    self.ann_infos[ann_id]['image_id'] = new_image_id

def edit_ann_class(self, ann_id, class_name=None, class_id=None):
    assert (class_name is not None) or (class_id is not None), 'Either class_name or class_id must be present'
    if class_name is not None:
        assert class_name in self.name_to_class_info
        class_id = self.name_to_label(class_name)
    else:
        assert class_id in self.id_to_class_info
        class_name = self.label_to_name(class_id)

    self.ann_infos[ann_id]['class_name'] = class_name
    self.ann_infos[ann_id]['class_id']   = class_id

def edit_ann_bbox(self, ann_id, bbox):
    bbox = self._prepare_bbox(bbox, self.ann_infos[ann_id]['image_id'])
    self.ann_infos[ann_id]['bbox'] = bbox
